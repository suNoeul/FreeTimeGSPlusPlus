# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn.functional import grid_sample
from tqdm import tqdm

from ftgspp.data.utils import MultiViewData
from ftgspp.data.utils.images import CAMERA_INDEX_WIDTH
from ftgspp.init import keyframe_indices, to_point_filename
from ftgspp.init.matcher import RoMaMatcher, create_roma_matcher
from ftgspp.utils import Config, PathLike
from ftgspp.utils.io import write_ply_points
from ftgspp.utils.math import chw

# Helpers


def k_nearest_cameras(
    w2cs: Tensor,  # (n, 4, 4)
    k: int,
) -> Tensor:
    c2ws = torch.inverse(w2cs)
    centers = c2ws[:, :3, 3]
    dists = torch.cdist(centers, centers)
    _, idxs = torch.topk(dists, k + 1, largest=False)

    return idxs[:, 1:]


def tensor_to_pil(tensor: Tensor):
    assert tensor.dtype == torch.uint8
    assert tensor.ndim == 3 and tensor.shape[-1] == 3
    return Image.fromarray(tensor.detach().cpu().numpy())


# Pipeline


def triangulate(
    uv,  # (cams, n, 2)
    projmat,  # (cams, 3, 4)
):
    # Multiple View Geometry (Hartley and Zisserman) Sec. 12.2
    p0, p1, p2 = torch.unbind(projmat, -2)
    x = torch.einsum("ij,ik->ijk", uv[..., 0], p2) - p0.unsqueeze(-2)
    y = torch.einsum("ij,ik->ijk", uv[..., 1], p2) - p1.unsqueeze(-2)
    eq = torch.cat([x, y]).transpose(0, 1)

    _, _, v = torch.svd(eq)

    return v[..., :3, 3] / v[..., 3:4, 3]


def matches_to_points(
    *,
    rgb_0: Tensor,
    pts_0: Tensor,
    pts_1: Tensor,
    intrinsic_0: Tensor,
    intrinsic_1: Tensor,
    w2c_0: Tensor,
    w2c_1: Tensor,
):
    image_shape = rgb_0.shape[:2]

    projmat0 = intrinsic_0 @ w2c_0[:3]
    projmat1 = intrinsic_1 @ w2c_1[:3]

    normalizer = torch.tensor(
        [
            [2 / image_shape[1], 0, -1],
            [0, 2 / image_shape[0], -1],
            [0, 0, 1],
        ]
    )

    xyz = triangulate(
        torch.stack(
            [
                pts_0 @ normalizer[:2, :2] + normalizer[:2, 2],
                pts_1 @ normalizer[:2, :2] + normalizer[:2, 2],
            ]
        ),
        torch.stack([normalizer @ projmat0, normalizer @ projmat1]),
    )

    cheiral = w2c_0[:3, 2] @ (xyz - w2c_0[:3, 3]).T
    valid = cheiral.squeeze() > 0
    valid *= torch.amax(xyz.abs(), dim=-1) < 1e3

    pts_0_norm = pts_0.clone()
    pts_0_norm[..., 0] /= image_shape[1]
    pts_0_norm[..., 1] /= image_shape[0]
    pts_0_norm = pts_0_norm * 2 - 1

    rgb = (
        grid_sample(
            chw(rgb_0).reshape(1, 3, *image_shape).float(),
            pts_0_norm.reshape(1, 1, -1, 2),
            align_corners=True,
            padding_mode="border",
        )
        .reshape(3, -1)
        .T
    )

    return xyz[valid], rgb[valid].byte()


def roma_points(
    matcher: RoMaMatcher,
    dataset: MultiViewData,  # (num_cameras,)
    num_nearest_cameras: int,
    matches_per_view: int,
    geometric_verification: bool,
    match_debug_path: Optional[PathLike] = None,
) -> tuple[Tensor, Tensor]:
    nearest_cameras = k_nearest_cameras(dataset.w2c, k=num_nearest_cameras)

    xyzs = []
    rgbs = []

    for cam_a in range(len(dataset)):
        for cam_b in nearest_cameras[cam_a].tolist():
            assert cam_a == dataset[cam_a].camera  # type: ignore
            assert cam_b == dataset[cam_b].camera  # type: ignore

            img_a = tensor_to_pil(dataset.rgb[cam_a])
            img_b = tensor_to_pil(dataset.rgb[cam_b])

            debug_path = None
            if match_debug_path is not None:
                debug_dir = Path(match_debug_path)
                img_a.save(debug_dir / f"{cam_a:0{CAMERA_INDEX_WIDTH}d}.png")
                img_b.save(debug_dir / f"{cam_b:0{CAMERA_INDEX_WIDTH}d}.png")
                debug_path = (
                    debug_dir
                    / f"{cam_a:0{CAMERA_INDEX_WIDTH}d}-{cam_b:0{CAMERA_INDEX_WIDTH}d}.png"
                )

            pts_a, pts_b = matcher.match_points(
                img_a=img_a,
                img_b=img_b,
                num=matches_per_view,
                debug_path=debug_path,
            )
            (cam_0, pts_0), (cam_1, pts_1) = sorted(((cam_a, pts_a), (cam_b, pts_b)))

            if geometric_verification:
                _fundmat, inliers = cv2.findFundamentalMat(
                    pts_0.numpy(),
                    pts_1.numpy(),
                    cv2.USAC_MAGSAC,
                    ransacReprojThreshold=3,
                    confidence=0.999,
                    maxIters=10000,
                )
                inliers = torch.tensor(inliers.squeeze().astype(np.bool_))
            else:
                inliers = torch.ones(len(pts_0), dtype=torch.bool)

            xyz, rgb = matches_to_points(
                rgb_0=dataset.rgb[cam_0],
                pts_0=pts_0[inliers],
                pts_1=pts_1[inliers],
                w2c_0=dataset.w2c[cam_0],
                w2c_1=dataset.w2c[cam_1],
                intrinsic_0=dataset.intrinsic[cam_0],
                intrinsic_1=dataset.intrinsic[cam_1],
            )

            xyzs.append(xyz)
            rgbs.append(rgb)

    return torch.cat(xyzs, 0), torch.cat(rgbs, 0)


def main(args: argparse.Namespace):
    config = Config.load(args.config)

    dataset: MultiViewData = MultiViewData.load_memmap(config.data.memmap_path)
    dataset = dataset.at(config.data.frames.into_slice())

    matcher = create_roma_matcher(config.init.roma, device="cuda")
    out_path = config.init.points_path
    out_path.mkdir(exist_ok=True, parents=True)

    matches_per_view = (
        config.init.num_points_per_frame
        // dataset.num_cameras
        // config.init.roma.num_nearest_cameras
    )

    idxs = keyframe_indices(dataset.num_frames, config.init.keyframe_stride)
    for idx in (bar := tqdm(idxs)):
        data: MultiViewData = dataset[idx]  # type: ignore

        xyz, rgb = roma_points(
            matcher=matcher,
            dataset=data,
            num_nearest_cameras=config.init.roma.num_nearest_cameras,
            matches_per_view=matches_per_view,
            geometric_verification=config.init.roma.geometric_verification,
            match_debug_path=None,
        )

        frame: int = data[0].frame  # type: ignore
        write_ply_points(
            out_path / to_point_filename(frame),
            xyz=xyz.cpu().numpy(),
            rgb=rgb.cpu().numpy(),
        )
        bar.set_postfix({"n": str(len(xyz))})

    torch.cuda.empty_cache()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(args)
