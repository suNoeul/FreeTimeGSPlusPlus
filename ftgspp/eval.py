# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import argparse
import math
import os
import pprint
from copy import deepcopy
from pathlib import Path
from typing import Callable, Mapping

import msgspec
import torch
from memfof import MEMFOF
from torch import Tensor
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

from ftgspp.data.utils import MultiViewData
from ftgspp.models.gaussians import Gaussians
from ftgspp.utils import Config, camera_indexer
from ftgspp.utils.math import chw, groupwise, hwc

type MetricFn = Callable[[Tensor, Tensor], float]


class MetricHistory(dict[str, list[float]]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mean(self) -> dict[str, float]:
        means: dict[str, float] = {}
        for name, history in self.items():
            valid = [x for x in history if not math.isnan(x)]
            means[name] = float("nan") if len(valid) == 0 else sum(valid) / len(valid)
        return means


class MetricTracker:
    _metric_fns: dict[str, MetricFn]
    _history: MetricHistory

    def __init__(self, metric_fns: Mapping[str, MetricFn]):
        self._metric_fns = dict(metric_fns)
        self._history = MetricHistory({k: [] for k in self._metric_fns})

    def __call__(self, pred: Tensor, gt: Tensor):
        for name, fn in self._metric_fns.items():
            metric = float(fn(pred, gt))
            self._history[name].append(metric)

    def history(self) -> MetricHistory:
        return deepcopy(self._history)


def make_lpips_metric(net_type: str) -> MetricFn:
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type=net_type, normalize=False
    ).cuda()

    def fn(pred: Tensor, gt: Tensor) -> float:
        # Match training-time LPIPS: normalize=False with [0, 1] inputs.
        pred_lpips = pred.clamp(0, 1)
        gt_lpips = gt.clamp(0, 1)
        return float(metric(pred_lpips, gt_lpips))

    return fn


@torch.inference_mode()
def eval_rgb(
    gs: Gaussians,
    eval_set: MultiViewData,
    metric_fns: Mapping[str, MetricFn],
) -> MetricHistory:
    tracker = MetricTracker(metric_fns)
    for batch in eval_set.flatten():
        batch: MultiViewData = batch.cuda()

        gt = batch.rgb.unsqueeze(0).float() / 255
        pred, _, _ = gs(
            t=batch.time,
            w2c=batch.w2c.unsqueeze(0),
            intrinsic=batch.intrinsic.unsqueeze(0),
            shape=(batch.height, batch.width),
        )

        tracker(chw(pred), chw(gt))

    return tracker.history()


@torch.inference_mode()
def eval_oflow(
    gs: Gaussians,
    eval_set: MultiViewData,
    metric_fns: Mapping[str, MetricFn],
) -> MetricHistory:
    flow_model = MEMFOF.from_pretrained("egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH")
    flow_model.eval().cuda()
    tracker = MetricTracker(metric_fns)

    for batch in tqdm(
        groupwise(eval_set.cuda(), 3),
        total=eval_set.num_frames - 2,
    ):
        prev, curr, next = batch

        frames = torch.stack([prev.rgb, curr.rgb, next.rgb]).cuda()
        data = torch.einsum("tnhwc->ntchw", frames)

        _flow_bwd, flow = flow_model(data)["flow"][-1].unbind(dim=1)

        flow_pred = gs.render_oflow(
            t0=curr.time,
            t1=next.time,
            w2c=curr.w2c,
            intrinsic=curr.intrinsic,
            shape=(eval_set.height, eval_set.width),
        )

        flow_gt = hwc(flow).contiguous()
        flow_pred = flow_pred.contiguous()
        tracker(flow_pred, flow_gt)
    return tracker.history()


def epe(pred: Tensor, gt: Tensor) -> float:
    assert pred.shape[-1] == gt.shape[-1] == 2
    l2 = torch.square(pred - gt).sum(dim=-1).sqrt()
    return float(l2.mean())


def n_pixel(pred: Tensor, gt: Tensor, n: float) -> float:
    assert pred.shape[-1] == gt.shape[-1] == 2
    l2 = torch.square(pred - gt).sum(dim=-1).sqrt()
    return 100 - 100 * float((l2 < n).float().mean())


@torch.no_grad()
def main(args: argparse.Namespace):
    config = Config.load(args.run_path / "config.toml")
    dataset = MultiViewData.load(config.data.memmap_path)
    eval_set = dataset.at(
        config.data.frames.into_slice(),
        camera_indexer(config.data.eval_cameras),
    )

    gs = Gaussians.load(args.run_path / args.filename)
    gs = gs.cuda().eval()

    metric_fns_rgb = {
        "psnr": PeakSignalNoiseRatio(data_range=1).cuda(),
        "lpips-alex": make_lpips_metric("alex"),
        "lpips-vgg": make_lpips_metric("vgg"),
        "ssim-1": StructuralSimilarityIndexMeasure(data_range=1).cuda(),
        "ssim-2": StructuralSimilarityIndexMeasure(data_range=2).cuda(),
    }
    metrics = eval_rgb(
        gs=gs,
        eval_set=eval_set,
        metric_fns=metric_fns_rgb,
    ).mean()
    metrics["dssim-1"] = (1 - metrics["ssim-1"]) / 2
    metrics["dssim-2"] = (1 - metrics["ssim-2"]) / 2
    metrics["size"] = os.path.getsize(args.run_path / args.filename)

    metric_fns_oflow = {
        "epe": epe,
        "1px-error": lambda pred, gt: n_pixel(pred, gt, 1),
        "3px-error": lambda pred, gt: n_pixel(pred, gt, 3),
        "5px-error": lambda pred, gt: n_pixel(pred, gt, 5),
    }
    metrics_oflow = eval_oflow(
        gs,
        eval_set,
        metric_fns_oflow,
    ).mean()

    metrics |= metrics_oflow

    pprint.pp(metrics)

    args.output.parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, "wb") as f:
        enc = msgspec.json.encode(metrics)
        enc = msgspec.json.format(enc)
        f.write(enc)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--filename", "-f", type=str, required=False, default="gaussians.pt"
    )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(args)
