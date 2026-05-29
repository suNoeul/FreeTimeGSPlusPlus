# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from pathlib import Path
from typing import Optional, cast

import torch
from torch import Tensor
from tqdm import tqdm

from ftgspp.data.utils import MultiViewData
from ftgspp.data.utils.images import FRAME_INDEX_WIDTH
from ftgspp.init.temporal import (
    TEMPORAL_OUTLIER_DISP_ALPHA,
    filter_temporal_outliers,
    temporal_ufm_velocities,
)
from ftgspp.models.gaussians import Gaussians
from ftgspp.utils.io import read_ply_points
from ftgspp.utils.math import knn_dist2_mean, nearest, rgb_to_sh


def to_point_filename(frame: int) -> str:
    return f"f{frame:0{FRAME_INDEX_WIDTH}d}.ply"


def keyframe_indices(num_frames: int, keyframe_stride: int) -> list[int]:
    frames = list(range(0, num_frames, keyframe_stride))
    if frames[-1] != num_frames - 1:
        frames.append(num_frames - 1)

    return frames


def init(
    *,
    dataset: MultiViewData,
    num_gaussians: int,
    keyframe_stride: int,
    points_path: Path,
    sh_degree: int,
    opacity: float,
    scale: float,
    duration: float,
    max_duration: float,
    num_velocity_nns: int,
    temporal_motion_adapted: bool = False,
    temporal_flow_path: Path = Path("_flow"),
    temporal_covis_thresh: float = 0.3,
) -> Gaussians:
    if temporal_motion_adapted:
        if not temporal_flow_path.exists():
            raise FileNotFoundError(f"temporal_flow_path does not exist: {temporal_flow_path}")
        override_stats: Optional[dict[str, int]] = {"num_override": 0, "num_total": 0}
    else:
        override_stats = None

    idxs = keyframe_indices(dataset.num_frames, keyframe_stride)
    keyframes = cast(list[int], dataset[idxs, 0].frame)  # type:ignore
    point_paths = [points_path / to_point_filename(frame) for frame in keyframes]

    idx_pairs = list(zip(idxs[:-1], idxs[1:]))
    point_path_pairs = list(zip(point_paths[:-1], point_paths[1:]))

    num_gaussians_per_frame = num_gaussians // len(point_paths)

    gs = Gaussians.empty(sh_degree=sh_degree, max_duration=max_duration)

    for (idx_0, idx_1), (path_0, path_1) in tqdm(
        zip(idx_pairs, point_path_pairs), total=len(point_path_pairs)
    ):
        xyz_0, rgb_0 = read_ply_points(path_0)
        xyz_1, _ = read_ply_points(path_1)

        t_0 = float(dataset[idx_0, 0].time)  # type:ignore
        t_1 = float(dataset[idx_1, 0].time)  # type:ignore

        gs = gs | init_dynamic(
            n=num_gaussians_per_frame,
            xyz=torch.tensor(xyz_0),
            rgb=torch.tensor(rgb_0).float() / 255,
            xyz_next=torch.tensor(xyz_1),
            time=t_0,
            delta_time=(t_1 - t_0),
            scale=scale,
            opacity=opacity,
            duration=duration,
            max_duration=max_duration,
            sh_degree=sh_degree,
            num_velocity_nns=num_velocity_nns,
            temporal_motion_adapted=temporal_motion_adapted,
            temporal_flow_path=temporal_flow_path,
            temporal_covis_thresh=temporal_covis_thresh,
            dataset=dataset,
            idx_0=idx_0,
            idx_1=idx_1,
            override_stats=override_stats,
        )

    # handle last frame separately

    xyz_last, rgb_last = read_ply_points(point_paths[-1])
    xyz_prev, _ = read_ply_points(point_paths[-2])
    t_last = float(dataset[idxs[-1], 0].time)  # type:ignore
    t_prev = float(dataset[idxs[-2], 0].time)  # type:ignore

    gs = gs | init_dynamic(
        n=num_gaussians_per_frame,
        xyz=torch.tensor(xyz_last),
        rgb=torch.tensor(rgb_last).float() / 255,
        xyz_next=torch.tensor(xyz_prev),
        time=t_last,
        delta_time=(t_prev - t_last),
        scale=scale,
        opacity=opacity,
        duration=duration,
        max_duration=max_duration,
        sh_degree=sh_degree,
        num_velocity_nns=num_velocity_nns,
        temporal_motion_adapted=temporal_motion_adapted,
        temporal_flow_path=temporal_flow_path,
        temporal_covis_thresh=temporal_covis_thresh,
        dataset=dataset,
        idx_0=idxs[-1],
        idx_1=idxs[-2],
        override_stats=override_stats,
    )

    if override_stats is not None and override_stats["num_total"] > 0:
        n_override = override_stats["num_override"]
        n_total = override_stats["num_total"]
        ratio = n_override / n_total
        print(
            f"[init][temporal] velocity override: {n_override}/{n_total} "
            f"({ratio:.2%}, fallback {(1 - ratio):.2%})"
        )

    return gs


def _as_knn_indices(idxs: Tensor) -> Tensor:
    idxs = torch.as_tensor(idxs, dtype=torch.long)
    if idxs.dim() == 1:
        idxs = idxs.unsqueeze(-1)
    return idxs


def knn_velocities(
    *,
    query_xyz: Tensor,
    ref_xyz: Tensor,
    delta_time: float,
    k: int,
) -> Tensor:
    k = max(1, min(k, len(ref_xyz)))
    nearest_idxs = _as_knn_indices(nearest(ref_xyz, query_xyz, k)).to(ref_xyz.device)
    velocities = (ref_xyz[nearest_idxs] - query_xyz.unsqueeze(-2)) / delta_time
    return velocities.mean(-2)


def init_dynamic(
    n: int,
    xyz: Tensor,
    xyz_next: Tensor,
    rgb: Tensor,
    time: float,
    delta_time: float,
    scale: float,
    opacity: float,
    duration: float,
    max_duration: float,
    sh_degree: int,
    num_velocity_nns: int,
    temporal_motion_adapted: bool = False,
    temporal_flow_path: Path = Path("_flow"),
    temporal_covis_thresh: float = 0.3,
    dataset: Optional[MultiViewData] = None,
    idx_0: Optional[int] = None,
    idx_1: Optional[int] = None,
    override_stats: Optional[dict[str, int]] = None,
) -> Gaussians:
    n = min(n, len(xyz), len(xyz_next))
    mask = torch.randint(0, len(xyz), (n,))

    means = xyz.clone()[mask]

    times = (torch.rand((n, 1), dtype=torch.float32) - 0.5) * delta_time + time

    # velocity
    velocities = knn_velocities(
        query_xyz=means,
        ref_xyz=xyz_next,
        delta_time=delta_time,
        k=num_velocity_nns,
    )
    dists2_local = torch.clamp(knn_dist2_mean(means), min=1e-7)
    local_spacing = torch.sqrt(dists2_local)

    if temporal_motion_adapted:
        if dataset is None or idx_0 is None or idx_1 is None:
            raise ValueError("temporal_motion_adapted requires dataset and frame indices")
        if not temporal_flow_path.exists():
            raise FileNotFoundError(f"temporal_flow_path does not exist: {temporal_flow_path}")
        temporal_vel, temporal_valid = temporal_ufm_velocities(
            query_xyz=means,
            dataset=dataset,
            idx_0=idx_0,
            idx_1=idx_1,
            delta_time=delta_time,
            cache_root=temporal_flow_path,
            covis_thresh=temporal_covis_thresh,
        )
        temporal_override = filter_temporal_outliers(
            temporal_vel=temporal_vel,
            temporal_valid=temporal_valid,
            local_spacing=local_spacing,
            delta_time=delta_time,
        )
        frame_0 = int(dataset[idx_0, 0].frame)  # type: ignore
        frame_1 = int(dataset[idx_1, 0].frame)  # type: ignore
        n_pre = int(temporal_valid.sum().item())
        n_post = int(temporal_override.sum().item())
        n_cut = n_pre - n_post
        print(
            f"[init][temporal][ufm][cutoff] f{frame_0:0{FRAME_INDEX_WIDTH}d}->"
            f"f{frame_1:0{FRAME_INDEX_WIDTH}d} "
            f"override={n_post}/{len(temporal_valid)} "
            f"({n_post / max(len(temporal_valid), 1):.2%}) "
            f"cutoff={n_cut}/{max(n_pre, 1)} alpha={TEMPORAL_OUTLIER_DISP_ALPHA:g}"
        )
        if override_stats is not None:
            override_stats["num_override"] += int(temporal_override.sum().item())
            override_stats["num_total"] += int(len(temporal_valid))
        velocities[temporal_override] = temporal_vel[temporal_override]

    # duration
    durations = torch.full((n, 1), duration, dtype=torch.float32)
    if max_duration == float("inf"):
        durations = torch.log(durations)
    else:
        durations = torch.logit(durations * 6 / max_duration)

    # quats
    quats = torch.zeros((n, 4))
    quats[:, 0] = 1

    # scales
    scales = scale * local_spacing.unsqueeze(-1).repeat(1, 3)
    scales = torch.log(scales)

    # opacities
    opacities = torch.full((n, 1), opacity, dtype=torch.float32)
    opacities = torch.logit(opacities)

    # colors
    sh = torch.zeros((n, (sh_degree + 1) ** 2, 3))
    sh[:, 0, :] = rgb_to_sh(rgb[mask])

    return Gaussians(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh_0=sh[:, 0:1, :],
        sh_n=sh[:, 1:, :],
        times=times,
        durations=durations,
        velocity_model=velocities,
        max_duration=max_duration,
    )
