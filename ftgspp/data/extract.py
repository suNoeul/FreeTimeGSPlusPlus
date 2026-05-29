# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import argparse
import asyncio
import json
import os
from asyncio import CancelledError, Semaphore, Server, StreamReader, StreamWriter
from asyncio.subprocess import Process
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from tqdm import tqdm

from ftgspp.data.utils.images import FRAME_INDEX_WIDTH, to_image_name
from ftgspp.utils import Config, num_workers


def _resolve_video_length(vidlen: int, frame_range: tuple[int, int | None]) -> int:
    if frame_range[1] is None:
        frame_range = (frame_range[0], vidlen)

    return cast(int, frame_range[1]) - frame_range[0]


@dataclass
class Extractor:
    video_path: Path
    out_path: Path
    num_processes: int
    compression_level: int = 0
    quality: int = 0
    frame_range: tuple[int, int | None] = (0, None)

    sem: Semaphore = field(init=False)
    videos: list[Path] = field(init=False)
    num_frames: int = field(init=False)
    processes: list[Process] = field(init=False)
    servers: list[Server] = field(init=False)
    bar: tqdm = field(init=False)

    def __post_init__(self):
        self.sem = Semaphore(self.num_processes)
        self.videos = sorted(self.video_path.glob("*.mp4"))

        async def _get_futures():
            return await asyncio.gather(*(video_length(video) for video in self.videos))

        video_lengths = asyncio.run(_get_futures())

        self.num_frames = sum(
            _resolve_video_length(vidlen, self.frame_range) for vidlen in video_lengths
        )
        self.processes = []
        self.servers = []
        self.bar = tqdm(disable=True)

    async def _launch(self, path, camera):
        async with self.sem:
            name = to_image_name(frame=f"%0{FRAME_INDEX_WIDTH}d", camera=camera)
            print(f"extracting {path}")
            s = await asyncio.start_server(
                self._handler, host="0.0.0.0", port=13337 + camera
            )
            self.servers.append(s)

            filter_trim = f"trim=start_frame={self.frame_range[0]}"
            if self.frame_range[1] is not None:
                filter_trim += f":end_frame={self.frame_range[1]}"

            p = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-nostats",
                "-nostdin",
                "-i",
                path,
                "-vf",
                f"{filter_trim},format=rgb24",
                "-vcodec",
                "libwebp",
                "-lossless",
                "1",
                "-compression_level",
                str(self.compression_level),
                "-quality",
                str(self.quality),
                "-progress",
                f"tcp://localhost:{13337 + camera}",
                "-start_number",
                str(self.frame_range[0]),
                str(self.out_path / name),
                preexec_fn=os.setpgrp,
            )

            self.processes.append(p)
            await p.wait()
            self.processes.remove(p)

    async def _handler(self, r: StreamReader, _w: StreamWriter):
        prev = 0

        while not r.at_eof():
            line = await r.readline()
            line = bytes.decode(line).strip()

            if line.startswith("frame="):
                frame = int(line.removeprefix("frame="))
                delta = frame - prev
                prev = frame

                self.bar.update(delta)

            if line == "progress=end":
                return

    async def start(self):
        self.bar = tqdm(total=self.num_frames)

        try:
            await asyncio.gather(
                *(self._launch(video, cam) for cam, video in enumerate(self.videos))
            )
        except CancelledError:
            print(self.processes)
            for p in self.processes:
                p.kill()
            await asyncio.gather(*(p.wait() for p in self.processes))
            for s in self.servers:
                s.close()


async def video_length(video) -> int:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream",
        "-of",
        "json",
        video,
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()

    return int(json.loads(stdout)["streams"][0]["nb_frames"])


def already_extracted(
    out_path: Path, num_cameras: int, frame_range: tuple[int, int | None]
) -> bool:
    start, end = frame_range
    if end is None:
        return False

    for camera in range(num_cameras):
        for frame in range(start, end):
            if not (out_path / to_image_name(frame=frame, camera=camera)).exists():
                return False

    return True


def main(args: argparse.Namespace):
    if args.input.suffix == ".toml":
        # Paths from file
        if args.output is not None:
            raise ValueError("explicit output paths cannot be used with a config file")
        config = Config.load(args.input)
        input_path = config.data.video_path
        output_path = config.data.extracted_path
        frame_range = (config.data.frames.start, config.data.frames.stop)
    else:
        # Paths from args
        if args.output is None:
            raise ValueError("output path must be specified")
        input_path = args.input
        output_path = args.output
        frame_range = (args.start, args.end)

    output_path.mkdir(exist_ok=True, parents=True)
    num_cameras = len(list(sorted(input_path.glob("*.mp4"))))
    if num_cameras == 0:
        raise FileNotFoundError(f"no mp4 files found in {input_path}")

    extractor = Extractor(
        video_path=input_path,
        out_path=output_path,
        num_processes=args.processes,
        compression_level=args.compression_level,
        quality=args.quality,
        frame_range=frame_range,
    )
    asyncio.run(extractor.start())


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="extracts frames of videos to images")
    parser.add_argument(
        "input",
        type=Path,
        help="input directory containing videos, or a configuration toml file",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="output directory to write extracted frames to, only valid with directory inputs",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="start frame index (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="end frame index (exclusive)",
    )
    parser.add_argument(
        "--processes",
        "-j",
        type=int,
        default=num_workers(),
        help="number of processes for video decoding",
    )
    parser.add_argument(
        "--compression-level",
        "-c",
        type=int,
        default=0,
        help="libwebp compression level, only affects lossless compression speed/ratio",
    )
    parser.add_argument(
        "--quality",
        "-q",
        type=int,
        default=20,
        help="libwebp compression quality, only affects lossless compression speed/ratio",
    )

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(args)
