# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class TrainState:
    grad2d: Tensor
    grad2d_vec: Tensor
    count: Tensor

    @classmethod
    def new(cls, num_gaussians: int):
        return cls(
            grad2d=torch.zeros(num_gaussians, device="cuda"),
            grad2d_vec=torch.zeros((num_gaussians, 2), device="cuda"),
            count=torch.zeros(num_gaussians, device="cuda"),
        )

    def update_(self, aux: dict[str, Any]):
        grad2d = aux["means2d"].grad.clone()
        grad2d[..., 0] *= aux["width"] / 2.0 * aux["n_cameras"]
        grad2d[..., 1] *= aux["height"] / 2.0 * aux["n_cameras"]

        self.grad2d.index_add_(0, aux["gaussian_ids"], grad2d.norm(dim=-1))
        self.grad2d_vec.index_add_(0, aux["gaussian_ids"], grad2d)
        self.count.index_add_(
            0,
            aux["gaussian_ids"],
            torch.ones_like(aux["gaussian_ids"], dtype=torch.float32),
        )

    def grad2d_acc(self) -> Tensor:
        return self.grad2d / self.count.clamp_min(1)

    def grad2d_coherence(self) -> Tensor:
        eps = torch.finfo(self.grad2d.dtype).eps
        return (self.grad2d_vec.norm(dim=-1) / self.grad2d.clamp_min(eps)).clamp(0, 1)

    def zero_(self):
        self.grad2d.zero_()
        self.grad2d_vec.zero_()
        self.count.zero_()
