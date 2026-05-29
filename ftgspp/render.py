# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import argparse
import math
from pathlib import Path
from typing import Literal, Sequence, cast, get_args

import cv2
import flip_evaluator
import imageio
import imageio.v3 as iio
import numpy as np
import torch
from memfof import MEMFOF
from torch import Tensor
from torchvision.utils import flow_to_image
from tqdm import tqdm

from ftgspp.data.utils import MultiViewData
from ftgspp.models.gaussians import Gaussians
from ftgspp.utils import Config
from ftgspp.utils.math import chw, ellipse_trajectory, groupwise, hwc, proj

ModeType = Literal[
    "rgb",
    "velocity-field",
    "flow",
    "flow-gt",
    "flow-comparison",
    "flow-comparison-dynmask",
    "error-map",
]
MODES: list[ModeType] = list(get_args(ModeType))


def intrinsic(shape: tuple[int, int], fov: float):
    h, w = shape
    cx, cy = w // 2, h // 2
    fx = w / 2 / math.tan(fov / 2)

    return torch.tensor(
        [[fx, 0, cx], [0, fx, cy], [0, 0, 1]], device="cuda", dtype=torch.float32
    )


def line3d(
    *,
    img: np.ndarray,
    start: Tensor,
    end: Tensor,
    w2c: Tensor,
    intrinsic: Tensor,
    image_shape: tuple[int, int],
    opacity: Tensor,
) -> np.ndarray:
    overlay = np.zeros((*image_shape, 4), dtype=np.uint8)

    w2c = w2c.unsqueeze(0)
    intrinsic = intrinsic.unsqueeze(0)
    start2d = proj(xyz=start, intrinsic=intrinsic, w2c=w2c, image_shape=image_shape)
    end2d = proj(xyz=end, intrinsic=intrinsic, w2c=w2c, image_shape=image_shape)

    start2d = start2d.squeeze(0).cpu().numpy().astype(np.int32)
    end2d = end2d.squeeze(0).cpu().numpy().astype(np.int32)

    opacity_np = opacity.flatten().cpu().numpy()
    visible = opacity_np > 1e-2
    for s, e, o in zip(start2d[visible], end2d[visible], opacity_np[visible].tolist()):
        overlay = cv2.arrowedLine(
            overlay,
            tuple(s.ravel()),
            tuple(e.ravel()),
            (255, 0, 0, int(float(o) * 255)),
            2,
            cv2.LINE_AA,
            tipLength=0.15,
        )

    return (
        img.astype(np.float32) * (1 - overlay[..., 3:] / 255)
        + overlay[..., :3].astype(np.float32) * overlay[..., 3:] / 255
    ).astype(np.uint8)


def require_camera(mode: ModeType, camera: int | None):
    if camera is not None:
        return

    if mode in {
        "error-map",
        "flow",
        "flow-gt",
        "flow-comparison",
        "flow-comparison-dynmask",
    }:
        raise ValueError(f"--camera is required when mode={mode!r}")


@torch.no_grad()
def render_rgb(
    *,
    gs: Gaussians,
    timestamps: Sequence[float],
    out: imageio.plugins.pyav.PyAVPlugin,
    w2cs: Tensor,
    shape: tuple[int, int],
    fov: float,
):
    for t, w2c in tqdm(
        zip(timestamps, w2cs),
        total=len(w2cs),
    ):
        pred, _, _ = gs(
            t=t,
            w2c=w2c.unsqueeze(0),
            intrinsic=intrinsic(shape, fov).unsqueeze(0),
            shape=shape,
        )

        img = pred.squeeze(0).mul(255).to(torch.uint8).detach().cpu().numpy()

        out.write_frame(img)


@torch.no_grad()
def render_velocity_field(
    *,
    gs: Gaussians,
    sample_rate: float,
    timestamps: Sequence[float],
    out: imageio.plugins.pyav.PyAVPlugin,
    w2cs: Tensor,
    shape: tuple[int, int],
    fov: float,
):
    dt = timestamps[1] - timestamps[0]
    n = int(len(gs) * sample_rate)
    selection = torch.randint(0, len(gs), (n,))

    for t, w2c in tqdm(
        zip(timestamps, w2cs),
        total=len(w2cs),
    ):
        pred, _, _ = gs(
            t=t,
            w2c=w2c.unsqueeze(0),
            intrinsic=intrinsic(shape, fov).unsqueeze(0),
            shape=shape,
        )

        img = pred.squeeze(0).mul(255).to(torch.uint8).detach().cpu().numpy()

        t = torch.tensor(t).to(gs.means).reshape(1, 1)

        line_start = gs.means_t(t)[selection]
        line_end = gs.means_t(t + dt)[selection]
        line_end = (line_end - line_start) * 10 + line_start

        img = line3d(
            img=img,
            start=line_start,
            end=line_end,
            intrinsic=intrinsic(shape, fov),
            w2c=w2c,
            image_shape=shape,
            opacity=gs.opacities_t(t)[selection],
        )

        out.write_frame(img)


@torch.no_grad()
def render_flip_errors(
    *,
    gs: Gaussians,
    out: imageio.plugins.pyav.PyAVPlugin,
    eval_set: MultiViewData,
):
    for batch in tqdm(eval_set):
        batch: MultiViewData
        pred, _, _ = gs(
            t=batch.time,
            w2c=batch.w2c.unsqueeze(0),
            intrinsic=batch.intrinsic.unsqueeze(0),
            shape=(batch.height, batch.width),
        )

        pred = pred.squeeze(0).detach().cpu().numpy()
        gt = batch.rgb.float().div(255).detach().cpu().numpy()
        errors, _, _ = flip_evaluator.evaluate(gt, pred, "LDR")
        errors = (errors * 255).astype(np.uint8)
        h, w = errors.shape[:2]
        errors = errors[: h - (h % 2), : w - (w % 2)]
        out.write_frame(errors)


@torch.no_grad()
def render_flow_gt(
    *,
    out: imageio.plugins.pyav.PyAVPlugin,
    dataset: MultiViewData,
):
    flow_model = MEMFOF.from_pretrained("egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH")
    flow_model.eval().cuda()

    with torch.inference_mode():
        for frames in tqdm(groupwise(dataset.rgb, 3), total=dataset.num_frames - 2):
            frames = torch.stack(frames).cuda()
            data = torch.einsum("tnhwc->ntchw", frames)

            _flow_bwd, flow = flow_model(data)["flow"][-1].unbind(dim=1)

            flow_img = hwc(flow_to_image(flow[0]))
            flow_img = flow_img.cpu().numpy()
            h, w = flow_img.shape[:2]
            flow_img = flow_img[: h - (h % 2), : w - (w % 2)]
            out.write_frame(flow_img)


@torch.no_grad()
def render_flow(
    *,
    gs: Gaussians,
    timestamps: Sequence[float],
    out: imageio.plugins.pyav.PyAVPlugin,
    w2cs: Tensor,
    shape: tuple[int, int],
    fov: float,
):
    t_pairs = zip(timestamps[:-1], timestamps[1:])
    total = max(len(timestamps) - 1, 0)

    for (t0, t1), w2c in tqdm(
        zip(t_pairs, w2cs[:-1]), total=total
    ):
        flow = gs.render_oflow(
            t0=t0,
            t1=t1,
            w2c=w2c.unsqueeze(0),
            intrinsic=intrinsic(shape, fov).unsqueeze(0),
            shape=shape,
        )[0]

        img = flow_to_image(chw(flow))
        out.write_frame(hwc(img).cpu().numpy())


@torch.no_grad()
def render_flow_comparison(
    *,
    gs: Gaussians,
    out: imageio.plugins.pyav.PyAVPlugin,
    dataset: MultiViewData,
):
    flow_model = MEMFOF.from_pretrained("egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH")
    flow_model.eval().cuda()

    with torch.inference_mode():
        for prev, curr, next in tqdm(
            groupwise(dataset.cuda(), 3), total=dataset.num_frames - 2
        ):
            frames = torch.stack([prev.rgb, curr.rgb, next.rgb]).cuda()
            data = torch.einsum("tnhwc->ntchw", frames)

            _flow_bwd, flow_gt = flow_model(data)["flow"][-1].unbind(dim=1)

            flow_pred = gs.render_oflow(
                t0=curr.time,
                t1=next.time,
                w2c=curr.w2c,
                intrinsic=curr.intrinsic,
                shape=(dataset.height, dataset.width),
            )

            flow_vis = torch.cat([chw(flow_pred[0]), flow_gt[0]], dim=2)

            flow_img = hwc(flow_to_image(flow_vis))
            flow_img = flow_img.cpu().numpy()
            h, w = flow_img.shape[:2]
            flow_img = flow_img[: h - (h % 2), : w - (w % 2)]
            out.write_frame(flow_img)


@torch.no_grad()
def render_flow_comparison_dynmask(
    *,
    gs: Gaussians,
    out: imageio.plugins.pyav.PyAVPlugin,
    dataset: MultiViewData,
    threshold: float,
):
    flow_model = MEMFOF.from_pretrained("egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH")
    flow_model.eval().cuda()

    with torch.inference_mode():
        for prev, curr, next in tqdm(
            groupwise(dataset.cuda(), 3), total=dataset.num_frames - 2
        ):
            frames = torch.stack([prev.rgb, curr.rgb, next.rgb]).cuda()
            data = torch.einsum("tnhwc->ntchw", frames)

            _flow_bwd, flow_gt = flow_model(data)["flow"][-1].unbind(dim=1)

            flow_pred = gs.render_oflow(
                t0=curr.time,
                t1=next.time,
                w2c=curr.w2c,
                intrinsic=curr.intrinsic,
                shape=(dataset.height, dataset.width),
            )

            flow_pred_chw = chw(flow_pred[0])
            flow_gt_chw = flow_gt[0]

            dynamic_mask = torch.linalg.norm(flow_gt_chw, dim=0) > threshold
            dynamic_mask = dynamic_mask.unsqueeze(0).to(flow_pred_chw.dtype)

            flow_pred_img = flow_to_image(flow_pred_chw).to(flow_pred_chw.dtype) * dynamic_mask
            flow_gt_img = flow_to_image(flow_gt_chw).to(flow_pred_chw.dtype) * dynamic_mask

            flow_vis = torch.cat([flow_pred_img, flow_gt_img], dim=2).to(torch.uint8)

            flow_img = hwc(flow_vis).cpu().numpy()
            h, w = flow_img.shape[:2]
            flow_img = flow_img[: h - (h % 2), : w - (w % 2)]
            out.write_frame(flow_img)


def main(args: argparse.Namespace):
    config = Config.load(args.run_path / "config.toml")
    mode = cast(ModeType, args.mode)
    require_camera(mode, args.camera)

    dataset = MultiViewData.load(config.data.memmap_path)
    h, w = (
        args.height or dataset.height,
        args.width or dataset.width,
    )
    h -= h % 2
    w -= w % 2

    if args.fov is None:
        fx = float(dataset.intrinsic[0, 0, 0, 0])
        fov = 2 * math.atan2(dataset.width, 2 * fx)
    else:
        fov = args.fov

    if args.camera is not None:
        w2cs = dataset[:, args.camera].w2c.cuda()  # type: ignore
    else:
        c2ws = torch.linalg.inv(dataset[0].w2c)  # type: ignore
        w2cs_avg = torch.linalg.inv(c2ws.mean(0))
        w2cs = ellipse_trajectory(
            w2cs_avg,
            dataset.num_frames,
            depth=args.depth or 10,
            rx=args.rx or 0.5,
            ry=args.ry or 0.5,
        ).cuda()

    gs = Gaussians.load(args.run_path / args.filename)
    gs.cuda().eval()

    args.output.parent.mkdir(exist_ok=True, parents=True)
    with iio.imopen(args.output, "w", plugin="pyav") as out:
        if out is not None:
            out.init_video_stream("libx264", fps=args.fps or config.data.fps)
            out._video_stream.options = {"crf": "20"}  # type: ignore

        match mode:
            case "rgb":
                render_rgb(
                    gs=gs,
                    timestamps=dataset[:, 0].time.cuda(),  # type: ignore
                    out=out,
                    w2cs=w2cs,
                    shape=(h, w),
                    fov=fov,
                )
            case "error-map":
                render_flip_errors(
                    gs=gs,
                    eval_set=dataset[:, args.camera].cuda(),  # type: ignore
                    out=out,
                )
            case "velocity-field":
                render_velocity_field(
                    gs=gs,
                    timestamps=dataset[:, 0].time.cuda(),  # type: ignore
                    out=out,
                    sample_rate=0.2,
                    w2cs=w2cs,
                    shape=(h, w),
                    fov=fov,
                )
            case "flow":
                render_flow(
                    gs=gs,
                    timestamps=dataset[:, args.camera].time.cuda(),  # type: ignore
                    out=out,
                    w2cs=w2cs,
                    shape=(h, w),
                    fov=fov,
                )
            case "flow-gt":
                render_flow_gt(
                    out=out,
                    dataset=dataset[:, args.camera : args.camera + 1],  # type: ignore
                )
                return
            case "flow-comparison":
                render_flow_comparison(
                    gs=gs,
                    dataset=dataset[:, args.camera : args.camera + 1],  # type: ignore
                    out=out,
                )
                return
            case "flow-comparison-dynmask":
                render_flow_comparison_dynmask(
                    gs=gs,
                    dataset=dataset[:, args.camera : args.camera + 1],  # type: ignore
                    out=out,
                    threshold=args.flow_dyn_threshold,
                )
                return


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--filename", "-f", type=str, required=False, default="gaussians.pt"
    )
    parser.add_argument("--camera", type=int, required=False)
    parser.add_argument("--width", type=int, required=False)
    parser.add_argument("--height", type=int, required=False)
    parser.add_argument("--fov", type=float, required=False)
    parser.add_argument("--rx", type=float, required=False)
    parser.add_argument("--ry", type=float, required=False)
    parser.add_argument("--depth", type=float, required=False)
    parser.add_argument("--fps", type=float, required=False)
    parser.add_argument(
        "--flow-dyn-threshold",
        type=float,
        required=False,
        default=1.0,
        help="Dynamic mask threshold in pixels for flow magnitude (used in flow-comparison-dynmask).",
    )
    parser.add_argument("--mode", choices=MODES, default="rgb")

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(args)
