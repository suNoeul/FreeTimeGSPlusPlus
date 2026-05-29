# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from math import isqrt
from typing import Optional, Self

import gsplat
import torch
from torch import Tensor, nn

from ftgspp.models.velocity import ExplicitVelocities, VelocityField
from ftgspp.utils import PathLike
from ftgspp.utils.math import proj


class Gaussians(nn.Module):
    means: nn.Parameter  # (n, 3)
    scales: nn.Parameter  # (n, 3)
    quats: nn.Parameter  # (n, 4)
    opacities: nn.Parameter  # (n, 1)
    sh_0: nn.Parameter  # (n, 1, 3)
    sh_n: nn.Parameter  # (n, (sh_degree + 1) ** 2 - 1, 3)
    times: nn.Parameter  # (n, 1)
    durations: nn.Parameter  # (n, 1)
    velocity_model: VelocityField | ExplicitVelocities

    marginal_gates: nn.Parameter  # (n, 3)
    sh_degree: int
    max_duration: float

    def __init__(
        self,
        means: Tensor,
        scales: Tensor,
        quats: Tensor,
        opacities: Tensor,
        sh_0: Tensor,
        sh_n: Tensor,
        times: Tensor,
        durations: Tensor,
        velocity_model: VelocityField | Tensor,
        max_duration: float,
        marginal_gates: Optional[Tensor] = None,
    ):
        super().__init__()

        self.means = nn.Parameter(means.float())
        self.scales = nn.Parameter(scales.float())
        self.quats = nn.Parameter(quats.float())
        self.opacities = nn.Parameter(opacities.float())
        self.sh_0 = nn.Parameter(sh_0.float())
        self.sh_n = nn.Parameter(sh_n.float())
        self.times = nn.Parameter(times.float())
        self.durations = nn.Parameter(durations.float())

        if isinstance(velocity_model, Tensor):
            velocity_model = nn.Parameter(velocity_model)
        self.velocity_model = velocity_model

        if marginal_gates is None:
            self.marginal_gates = nn.Parameter(torch.full((len(self), 1), -1).float())
        else:
            self.marginal_gates = nn.Parameter(marginal_gates)

        self.sh_degree = isqrt(sh_n.shape[1] + 1) - 1
        self.max_duration = max_duration

    def means_t(self, t: float | Tensor) -> Tensor:
        return self.means + (t - self.times) * self.velocities_t(t)

    def velocities_t(self, t: float | Tensor):
        match self.velocity_model:
            case VelocityField():
                return self.velocity_model(self.means, t)
            case ExplicitVelocities():
                return self.velocity_model

    def gate(self):
        return torch.sigmoid(20 * self.marginal_gates)

    def temporal_opacity(self, t: float | Tensor):
        gate = self.gate()
        if self.max_duration == float("inf"):
            tscale = torch.exp(self.durations)
        else:
            tscale = self.max_duration / 6 * torch.sigmoid(self.durations)
        return gate + (1 - gate) * torch.exp(-0.5 * ((t - self.times) / tscale) ** 2)

    def opacities_t(
        self,
        t: float | Tensor,
    ) -> Tensor:
        return self.opacities.sigmoid() * self.temporal_opacity(t)

    def forward(
        self,
        t: float | Tensor,  # () or (1,) or (1, 1)
        w2c: Tensor,  # (B, 4, 4)
        intrinsic: Tensor,  # (B, 3, 3)
        shape: tuple[int, int],
        clamp: bool = True,
        sh_degree: Optional[int] = None,
    ):
        if not isinstance(t, Tensor):
            t = torch.tensor(t).to(self.means)
        t = t.view(1, 1)

        if sh_degree is None:
            sh_degree = self.sh_degree

        image, alpha, meta = gsplat.rasterization(
            means=self.means_t(t),
            quats=self.quats,
            scales=self.scales.exp(),
            opacities=self.opacities_t(t).squeeze(-1),
            colors=torch.cat([self.sh_0, self.sh_n], dim=1),
            viewmats=w2c,
            Ks=intrinsic,
            sh_degree=sh_degree,
            width=shape[1],
            height=shape[0],
        )
        if clamp:
            image = image.clone().clamp(0, 1)

        return image, alpha, meta

    def render_oflow(
        self,
        t0: float | Tensor,  # () or (1,) or (1, 1)
        t1: float | Tensor,  # () or (1,) or (1, 1)
        w2c: Tensor,  # (B, 4, 4)
        intrinsic: Tensor,  # (B, 3, 3)
        shape: tuple[int, int],
    ):
        start = self.means_t(t0)
        end = self.means_t(t1)

        start2d = proj(xyz=start, intrinsic=intrinsic, w2c=w2c, image_shape=shape)
        end2d = proj(xyz=end, intrinsic=intrinsic, w2c=w2c, image_shape=shape)
        delta = end2d - start2d

        flow, _, _ = gsplat.rasterization(
            means=start,
            quats=self.quats,
            scales=self.scales.exp(),
            opacities=self.opacities_t(t0).squeeze(-1),
            colors=delta,
            viewmats=w2c,
            Ks=intrinsic,
            sh_degree=None,
            width=shape[1],
            height=shape[0],
            packed=True,
        )

        return flow

    def mask(self, mask: Tensor) -> Self:
        match self.velocity_model:
            case VelocityField():
                v = self.velocity_model
            case ExplicitVelocities():
                v = self.velocity_model[mask]

        return self.__class__(
            means=self.means[mask],
            scales=self.scales[mask],
            quats=self.quats[mask],
            opacities=self.opacities[mask],
            sh_0=self.sh_0[mask],
            sh_n=self.sh_n[mask],
            times=self.times[mask],
            durations=self.durations[mask],
            velocity_model=v,
            marginal_gates=self.marginal_gates[mask],
            max_duration=self.max_duration,
        )

    @classmethod
    def empty(cls, sh_degree: int, max_duration: float = float("inf")) -> Self:
        return cls(
            means=torch.empty((0, 3)),
            scales=torch.empty((0, 3)),
            quats=torch.empty((0, 4)),
            opacities=torch.empty((0, 1)),
            sh_0=torch.empty((0, 1, 3)),
            sh_n=torch.empty((0, (sh_degree + 1) ** 2 - 1, 3)),
            times=torch.empty((0, 1)),
            durations=torch.empty((0, 1)),
            velocity_model=torch.empty((0, 3)),
            marginal_gates=torch.empty((0, 1)),
            max_duration=max_duration,
        )

    def __or__(self, other: Self) -> Self:
        if len(self) == 0:
            max_duration = other.max_duration
        elif len(other) == 0:
            max_duration = self.max_duration
        elif self.max_duration != other.max_duration:
            raise ValueError(
                "cannot merge Gaussians with different max_duration "
                f"({self.max_duration} != {other.max_duration})"
            )
        else:
            max_duration = self.max_duration

        match self.velocity_model, other.velocity_model:
            case VelocityField(), VelocityField():
                v = self.velocity_model
            case ExplicitVelocities(), ExplicitVelocities():
                v = torch.cat([self.velocity_model, other.velocity_model])
            case _:
                raise ValueError

        return self.__class__(
            means=torch.cat([self.means, other.means]),
            scales=torch.cat([self.scales, other.scales]),
            quats=torch.cat([self.quats, other.quats]),
            opacities=torch.cat([self.opacities, other.opacities]),
            sh_0=torch.cat([self.sh_0, other.sh_0]),
            sh_n=torch.cat([self.sh_n, other.sh_n]),
            times=torch.cat([self.times, other.times]),
            durations=torch.cat([self.durations, other.durations]),
            velocity_model=v,
            marginal_gates=torch.cat([self.marginal_gates, other.marginal_gates]),
            max_duration=max_duration,
        )

    def __len__(self) -> int:
        return len(self.means)

    def save(self, path: PathLike):
        torch.save(self, path)

    @classmethod
    def load(cls, path: PathLike) -> Self:
        return torch.load(path, weights_only=False)
