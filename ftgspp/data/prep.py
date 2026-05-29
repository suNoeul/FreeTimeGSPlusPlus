# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import argparse
from pathlib import Path

from ftgspp.data.utils.multiview import MultiViewData
from ftgspp.utils import Config


def main(args: argparse.Namespace):
    config = Config.load(args.config)

    MultiViewData.new(
        image_path=config.data.extracted_path,
        out_path=config.data.memmap_path,
        colmap_path=config.data.colmap_path,
        fps=config.data.fps,
        scale=config.data.scale,
        frame_range=config.data.frames,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(args)
