# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import math
from typing import Callable, Literal, Mapping

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ftgspp.utils.math import clamp


class OptimizerCollection(Mapping[str, Optimizer]):
    _optimizers: dict[str, Optimizer]

    def __init__(self, optimizers: Mapping[str, Optimizer]):
        self._optimizers = dict(optimizers)

    def step(self, *args, **kwargs):
        for opt in self._optimizers.values():
            opt.step(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = True):
        for opt in self._optimizers.values():
            opt.zero_grad(set_to_none)

    def __getitem__(self, key: str) -> Optimizer:
        return self._optimizers[key]

    def __iter__(self):
        return iter(self._optimizers)

    def __len__(self) -> int:
        return len(self._optimizers)

    def asdict(self) -> dict[str, Optimizer]:
        return self._optimizers


class LRSchedulerCollection(dict[str, LRScheduler]):
    def step(self):
        for lr_scheduler in self.values():
            lr_scheduler.step()


class FunctionalLR(LRScheduler):
    fn: Callable[[float], float]
    total_steps: int
    multiplicative: bool

    def __init__(
        self,
        optimizer: Optimizer,
        fn: Callable[[float], float],
        total_steps: int,
        multiplicative: bool = True,
        last_epoch: int = -1,
    ):
        self.fn = fn
        self.total_steps = total_steps
        self.multiplicative = multiplicative
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        progress = self.last_epoch / self.total_steps
        return [
            self.fn(progress) * (base_lr if self.multiplicative else 1)
            for base_lr in self.base_lrs
        ]


class GammaDensityLR(LRScheduler):
    total_steps: int
    peak: float
    multiplicative: bool

    def __init__(
        self,
        optimizer: Optimizer,
        peak: float,
        total_steps: int,
        multiplicative: bool = True,
        last_epoch: int = -1,
    ):
        self.total_steps = total_steps
        self.peak = peak
        self.multiplicative = multiplicative
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        progress = self.last_epoch / self.total_steps / self.peak
        schedule = progress * math.exp(1 - progress)

        return [
            schedule * (base_lr if self.multiplicative else 1)
            for base_lr in self.base_lrs
        ]


class CosineLR(LRScheduler):
    total_steps: int
    saturation_step: int
    multiplicative: bool
    mode: Literal["decay", "rampup"]

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
        saturation_step: int,
        mode: Literal["decay", "rampup"],
        multiplicative: bool = True,
        last_epoch: int = -1,
    ):
        self.total_steps = total_steps
        self.saturation_step = saturation_step
        self.mode = mode
        self.multiplicative = multiplicative
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        sign = 1 if self.mode == "decay" else -1
        progress = self.last_epoch / self.saturation_step
        schedule = (1 + sign * math.cos(math.pi * clamp(progress, 0, 1))) / 2

        return [
            schedule * (base_lr if self.multiplicative else 1)
            for base_lr in self.base_lrs
        ]
