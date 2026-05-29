# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ftgspp.data.utils import MultiViewData
from ftgspp.data.utils.images import CAMERA_INDEX_WIDTH, FRAME_INDEX_WIDTH

TEMPORAL_OUTLIER_DISP_ALPHA = 3.0


def to_flow_pair_dir(frame_0: int, frame_1: int) -> str:
    return f"f{frame_0:0{FRAME_INDEX_WIDTH}d}_f{frame_1:0{FRAME_INDEX_WIDTH}d}"


def to_flow_cam_filename(camera: int) -> str:
    return f"c{camera:0{CAMERA_INDEX_WIDTH}d}.npz"


def project_points(
    xyz: Tensor,
    w2c: Tensor,
    intrinsic: Tensor,
    image_shape: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor]:
    xyz_cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = xyz_cam[:, 2]

    uv_h = xyz_cam @ intrinsic.T
    uv = uv_h[:, :2] / torch.clamp(uv_h[:, 2:3], min=1e-7)

    h, w = image_shape
    valid = z > 1e-6
    valid &= uv[:, 0] >= 0
    valid &= uv[:, 0] < w
    valid &= uv[:, 1] >= 0
    valid &= uv[:, 1] < h

    return uv, z, valid


def sample_features(image: Tensor, uv: Tensor) -> Tensor:
    if image.dim() == 2:
        image = image.unsqueeze(-1)

    if len(uv) == 0:
        return torch.empty((0, image.shape[-1]), dtype=image.dtype, device=image.device)

    h, w = image.shape[:2]
    uv_norm = uv.clone()
    uv_norm[:, 0] = uv_norm[:, 0] / max(w - 1, 1) * 2 - 1
    uv_norm[:, 1] = uv_norm[:, 1] / max(h - 1, 1) * 2 - 1

    sampled = torch.nn.functional.grid_sample(
        image.permute(2, 0, 1).unsqueeze(0),
        uv_norm.view(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.view(image.shape[-1], -1).T


def backproject_points(
    uv: Tensor,
    depth: Tensor,
    intrinsic: Tensor,
    w2c: Tensor,
) -> Tensor:
    uv_h = torch.cat([uv, torch.ones((len(uv), 1), dtype=uv.dtype, device=uv.device)], dim=-1)
    xyz_cam = (torch.inverse(intrinsic) @ uv_h.T).T * depth.unsqueeze(-1)
    c2w = torch.inverse(w2c)
    xyz_world = xyz_cam @ c2w[:3, :3].T + c2w[:3, 3]
    return xyz_world


def temporal_ufm_velocities(
    *,
    query_xyz: Tensor,
    dataset: MultiViewData,
    idx_0: int,
    idx_1: int,
    delta_time: float,
    cache_root: Path,
    covis_thresh: float,
) -> tuple[Tensor, Tensor]:
    vel_sum = torch.zeros_like(query_xyz)
    weight_sum = torch.zeros((len(query_xyz),), dtype=torch.float32, device=query_xyz.device)

    frame_0 = int(dataset[idx_0, 0].frame)  # type: ignore
    frame_1 = int(dataset[idx_1, 0].frame)  # type: ignore
    pair_dir = cache_root / to_flow_pair_dir(frame_0, frame_1)
    if not pair_dir.exists():
        raise FileNotFoundError(f"flow pair directory not found: {pair_dir}")

    total_projected = 0
    total_covis = 0
    total_final = 0
    cams_with_cache = 0
    cam_details: list[tuple[int, int, int, int]] = []
    # tuple: (cam, projected, covis, final)

    for cam in range(dataset.num_cameras):
        cache_path = pair_dir / to_flow_cam_filename(cam)
        if not cache_path.exists():
            raise FileNotFoundError(f"flow file not found: {cache_path}")
        cams_with_cache += 1

        with np.load(cache_path) as npz:
            flow = torch.from_numpy(np.asarray(npz["flow"])).to(query_xyz).float()
            covis = torch.from_numpy(np.asarray(npz["covis"])).to(query_xyz).float()

        if flow.dim() != 3 or flow.shape[-1] != 2:
            raise ValueError(f"Invalid flow shape in {cache_path}: {tuple(flow.shape)}")

        if covis.dim() == 3 and covis.shape[-1] == 1:
            covis = covis[..., 0]
        if covis.dim() != 2:
            raise ValueError(f"Invalid covis shape in {cache_path}: {tuple(covis.shape)}")

        h, w = int(flow.shape[0]), int(flow.shape[1])
        if covis.shape != (h, w):
            raise ValueError(
                f"Flow/covis size mismatch in {cache_path}: "
                f"{tuple(flow.shape)} vs {tuple(covis.shape)}"
            )

        if (h, w) != (dataset.height, dataset.width):
            flow_resized = torch.nn.functional.interpolate(
                flow.permute(2, 0, 1).unsqueeze(0),
                size=(dataset.height, dataset.width),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0).permute(1, 2, 0)
            flow_resized[..., 0] *= dataset.width / max(w, 1)
            flow_resized[..., 1] *= dataset.height / max(h, 1)
            flow = flow_resized
            covis = torch.nn.functional.interpolate(
                covis.reshape(1, 1, h, w),
                size=(dataset.height, dataset.width),
                mode="bilinear",
                align_corners=True,
            ).reshape(dataset.height, dataset.width)

        uv, depth, valid = project_points(
            query_xyz,
            w2c=dataset.w2c[idx_0, cam].to(query_xyz),
            intrinsic=dataset.intrinsic[idx_0, cam].to(query_xyz),
            image_shape=(dataset.height, dataset.width),
        )
        valid_idxs = valid.nonzero(as_tuple=False).flatten()
        if len(valid_idxs) == 0:
            cam_details.append((cam, 0, 0, 0))
            continue

        n_projected = int(len(valid_idxs))
        src = uv[valid_idxs]
        src_depth = depth[valid_idxs]

        flow_src = sample_features(flow, src)
        covis_src = sample_features(covis, src).squeeze(-1)
        dst = src + flow_src

        keep = covis_src >= covis_thresh
        keep &= dst[:, 0] >= 0
        keep &= dst[:, 0] < dataset.width
        keep &= dst[:, 1] >= 0
        keep &= dst[:, 1] < dataset.height
        n_covis = int(keep.sum().item())
        if not bool(keep.any()):
            cam_details.append((cam, n_projected, n_covis, 0))
            total_projected += n_projected
            total_covis += n_covis
            continue

        idxs = valid_idxs[keep]
        src = src[keep]
        dst = dst[keep]
        src_depth = src_depth[keep]

        w2c_0 = dataset.w2c[idx_0, cam].to(query_xyz)
        w2c_1 = dataset.w2c[idx_1, cam].to(query_xyz)
        intr_0 = dataset.intrinsic[idx_0, cam].to(query_xyz)
        intr_1 = dataset.intrinsic[idx_1, cam].to(query_xyz)

        xyz_src = backproject_points(src, src_depth, intr_0, w2c_0)
        xyz_dst = backproject_points(dst, src_depth, intr_1, w2c_1)
        vel = (xyz_dst - xyz_src) / delta_time

        weight = sample_features(covis, src).squeeze(-1).clamp(min=1e-6)

        vel_sum.index_add_(0, idxs, vel * weight.unsqueeze(-1))
        weight_sum.index_add_(0, idxs, weight)
        n_final = int(len(idxs))

        total_projected += n_projected
        total_covis += n_covis
        total_final += n_final
        cam_details.append((cam, n_projected, n_covis, n_final))

    valid = weight_sum > 0
    out = torch.zeros_like(query_xyz)
    out[valid] = vel_sum[valid] / weight_sum[valid].unsqueeze(-1)
    num_override = int(valid.sum().item())
    denom = max(len(query_xyz), 1)
    print(
        f"[init][temporal][ufm] f{frame_0:0{FRAME_INDEX_WIDTH}d}->"
        f"f{frame_1:0{FRAME_INDEX_WIDTH}d} "
        f"cache_cams={cams_with_cache}/{dataset.num_cameras} "
        f"override={num_override}/{len(query_xyz)} ({num_override / denom:.2%}) "
        f"proj={total_projected} covis={total_covis} final={total_final}"
    )
    if cam_details:
        top = sorted(cam_details, key=lambda x: x[3], reverse=True)[:5]
        top_s = " ".join(
            f"c{cam:03d}:{final}/{proj}"
            for cam, proj, _covis, final in top
            if proj > 0
        )
        if top_s:
            print(f"[init][temporal][ufm] top_cam_override {top_s}")
    return out, valid


def filter_temporal_outliers(
    *,
    temporal_vel: Tensor,
    temporal_valid: Tensor,
    local_spacing: Tensor,
    delta_time: float,
) -> Tensor:
    disp = torch.linalg.norm(temporal_vel, dim=-1) * abs(delta_time)
    disp_limit = TEMPORAL_OUTLIER_DISP_ALPHA * local_spacing
    return temporal_valid & (disp <= disp_limit)
