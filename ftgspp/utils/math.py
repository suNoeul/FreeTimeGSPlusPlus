# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from collections import deque
from itertools import islice
from typing import Callable, Iterable, Iterator, Optional

import gsplat
import numpy as np
import scipy
import scipy.spatial
import torch
from torch import Tensor, nn

# image ops


def tensor_to_img(img: Tensor) -> np.ndarray:
    return img.detach().clip(0, 1).mul(255).to(torch.uint8).cpu().numpy()


def chw(hwc: Tensor) -> Tensor:
    return hwc.permute(*range(hwc.dim() - 3), -1, -3, -2)


def hwc(chw: Tensor) -> Tensor:
    return chw.permute(*range(chw.dim() - 3), -2, -1, -3)


# spatial


def knn_dist2_mean(
    xyz: Tensor, xyz_other: Optional[Tensor] = None, k: int = 3
) -> torch.Tensor:
    if xyz_other is None:
        xyz_other = xyz.clone()

    xyz_np = xyz.detach().cpu().numpy()
    xyz_other_np = xyz_other.detach().cpu().numpy()

    kdtree = scipy.spatial.KDTree(xyz_np)
    dists_np, _ = kdtree.query(xyz_other_np, k=k)
    dists = torch.tensor(dists_np)
    dists2_mean = torch.mean(dists**2, dim=1)

    return dists2_mean.to(xyz)


def nearest(x0: Tensor, x1: Tensor, k: int = 1) -> Tensor:
    x0_np = x0.detach().cpu().numpy()
    x1_np = x1.detach().cpu().numpy()

    kdtree = scipy.spatial.KDTree(x0_np)
    _, idxs = kdtree.query(x1_np, k=k)

    return torch.tensor(idxs)


def scene_extent(w2c: Tensor) -> float:
    c2w = torch.inverse(w2c)
    positions = c2w[:, :3, 3]
    center = positions.mean(dim=0)
    dists = torch.norm(positions - center, dim=1)
    return torch.max(dists).item()


# views


def look_at(
    look: Tensor,  # (N, 3)
    z: Tensor = torch.tensor([[0, -1, 0]]),
) -> Tensor:  # (N, 3, 3)
    z = z.to(look)

    look = nn.functional.normalize(look)
    right = nn.functional.normalize(torch.cross(look, z, dim=1))
    up = nn.functional.normalize(torch.cross(right, look, dim=1))

    return torch.stack([right, -up, look], dim=2)


def ellipse_trajectory(
    w2c_center: Tensor,
    n: int,
    *,
    depth: float,
    rx: float,
    ry: float,
    periods: int = 2,
) -> torch.Tensor:
    c2w_center = torch.inverse(w2c_center)

    right, up, forward, center = c2w_center[:3].unsqueeze(0).unbind(2)
    focus = center + forward * depth

    t = torch.linspace(0, 1, n).to(w2c_center)
    t *= 2 * periods * torch.pi

    delta_u = rx * right * torch.sin(t).unsqueeze(1)
    delta_v = ry * up * torch.cos(t).unsqueeze(1)

    position = c2w_center[:3, 3].unsqueeze(0) + delta_u + delta_v  # (N, 3)
    look = focus - position
    rotation = look_at(look, z=-up)

    c2ws = torch.tile(torch.eye(4), (n, 1, 1))
    c2ws[:, :3, :3] = rotation
    c2ws[:, :3, 3] = position
    w2cs = torch.inverse(c2ws)

    return w2cs.to(w2c_center)


def proj(xyz: Tensor, w2c: Tensor, intrinsic: Tensor, image_shape: tuple[int, int]):
    covars = torch.eye(3, 3).unsqueeze(0).expand(len(xyz), 3, 3).to(xyz)
    xyz_cam, covars_proj = gsplat.world_to_cam(means=xyz, covars=covars, viewmats=w2c)
    uv, _ = gsplat.proj(
        means=xyz_cam,
        covars=covars_proj,
        Ks=intrinsic,
        height=image_shape[0],
        width=image_shape[1],
    )

    return uv


# spherical harmonics


SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb):
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh):
    return sh * SH_C0 + 0.5


# data


def rmoutliers(
    x: Tensor,
    p: tuple[float, float],
    reduce_fn: Callable[[Tensor], Tensor] = lambda x: x,
) -> tuple[Tensor, Tensor]:
    lo, hi = p

    reduced = reduce_fn(x)
    assert reduced.dim() == 1
    hi_val = reduced.quantile(hi)
    lo_val = reduced.quantile(lo)

    mask = (reduced > lo_val) & (reduced < hi_val)

    return x[mask], mask


# collection


def groupwise[T](iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    it = iter(iterable)
    accum = deque(islice(it, n - 1), maxlen=n)
    for x in it:
        accum.append(x)
        yield tuple(accum)


# math


def clamp(x: float, minval: float, maxval: float) -> float:
    return min(max(x, minval), maxval)


def normalize_minmax(
    x: torch.Tensor, reduce_dims: tuple[int, ...] = ()
) -> torch.Tensor:
    return (x - x.amin(reduce_dims)) / (x.amax(reduce_dims) - x.amin(reduce_dims))


def tv_2d(x: Tensor) -> Tensor:
    """2D total variation, expects (N, H, W, ...) input"""
    tv_x = torch.mean(torch.square(x[:, :, 1:, ...] - x[:, :, :-1, ...]))
    tv_y = torch.mean(torch.square(x[:, 1:, :, ...] - x[:, :-1, :, ...]))
    return tv_x + tv_y
