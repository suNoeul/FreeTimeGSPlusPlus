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
from torch import Tensor
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from ftgspp.data.utils import MultiViewData
from ftgspp.models.gaussians import Gaussians
from ftgspp.utils import Config, camera_indexer
from ftgspp.utils.math import chw

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


def make_lpips_metric(net_type: str, normalize: bool = False) -> MetricFn:
    # normalize=False matches training-time LPIPS with [0, 1] inputs (our
    # reporting convention); normalize=True is measured alongside for reference.
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type=net_type, normalize=normalize
    ).cuda()

    def fn(pred: Tensor, gt: Tensor) -> float:
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
        "lpips-alex": make_lpips_metric("alex", normalize=False),
        "lpips-vgg": make_lpips_metric("vgg", normalize=False),
        "lpips-alex-norm": make_lpips_metric("alex", normalize=True),
        "lpips-vgg-norm": make_lpips_metric("vgg", normalize=True),
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
