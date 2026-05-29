# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from typing import Callable


class Pipeline:
    stages: dict[str, Callable] = {}

    def __init_subclass__(cls):
        cls.stages = {}
        for method in cls.__dict__.values():
            if hasattr(method, "_stage_name"):
                cls.stages[method._stage_name] = method

    @classmethod
    def stage(cls, name: str):
        def decorator(f):
            f._stage_name = name
            return f

        return decorator
