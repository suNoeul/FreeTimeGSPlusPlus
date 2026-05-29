# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class TrainState:
    grad2d: Tensor
    count: Tensor

    @classmethod
    def new(cls, num_gaussians: int):
        return cls(
            grad2d=torch.zeros(num_gaussians, device="cuda"),
            count=torch.zeros(num_gaussians, device="cuda"),
        )

    def update_(self, aux: dict[str, Any]):
        grad2d = aux["means2d"].grad.clone()
        grad2d[..., 0] *= aux["width"] / 2.0 * aux["n_cameras"]
        grad2d[..., 1] *= aux["height"] / 2.0 * aux["n_cameras"]

        self.grad2d.index_add_(0, aux["gaussian_ids"], grad2d.norm(dim=-1))
        self.count.index_add_(
            0,
            aux["gaussian_ids"],
            torch.ones_like(aux["gaussian_ids"], dtype=torch.float32),
        )

    def grad2d_acc(self) -> Tensor:
        return self.grad2d / self.count.clamp_min(1)

    def zero_(self):
        self.grad2d.zero_()
        self.count.zero_()
