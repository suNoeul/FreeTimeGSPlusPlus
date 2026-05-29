# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import random
from typing import Protocol


class SizedDataset[T](Protocol):
    def __getitem__(self, index) -> T: ...
    def __getitems__(self, index) -> T: ...
    def __len__(self) -> int: ...


class FastInfiniteLoader[T]:
    dataset: SizedDataset[T]
    batch_size: int | None

    def __init__(self, dataset: SizedDataset[T], batch_size: int | None):
        self.dataset = dataset
        self.batch_size = batch_size

    def __next__(self):
        if self.batch_size is not None:
            indices = random.choices(range(len(self.dataset)), k=self.batch_size)
        else:
            indices = random.randint(0, len(self.dataset) - 1)
        return self.dataset[indices]
