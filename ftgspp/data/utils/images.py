# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from os import PathLike
from pathlib import Path
from typing import Sequence

from ftgspp.utils import Interval

FRAME_INDEX_WIDTH = 6
CAMERA_INDEX_WIDTH = 3
SUFFIX = ".webp"


def to_image_name(frame: int | str, camera: int | str, suffix: str = SUFFIX) -> str:
    if isinstance(frame, int):
        frame = f"{frame:0{FRAME_INDEX_WIDTH}d}"
    if isinstance(camera, int):
        camera = f"{camera:0{CAMERA_INDEX_WIDTH}d}"

    return f"c{camera}-f{frame}{suffix}"


def scan_images(
    path: PathLike, frame_range: Interval
) -> tuple[Sequence[int], Sequence[int], str]:
    path = Path(path)
    first_frames = list(
        path.glob(to_image_name(frame=frame_range.start, camera="*", suffix=".*"))
    )
    num_total_cameras = len(first_frames)
    frames = frame_range.into_range()
    cameras = range(num_total_cameras)

    return frames, cameras, first_frames[0].suffix
