# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import argparse
import logging
import time
import urllib.parse
from pathlib import Path

import rerun as rr

from ftgspp.train import Trainer
from ftgspp.utils import Config

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args: argparse.Namespace):
    config = Config.load(args.config)

    if args.rerun is not None:
        rr.serve_web_viewer(open_browser=False)
        rr.init("ftgspp")
        if args.rerun == "web":
            grpc_url = rr.serve_grpc()
            logger.info("serving rerun on http://localhost:9090/?url=%s", urllib.parse.quote(grpc_url))
        else:
            rr.connect_grpc(args.rerun)

    trainer = Trainer(config=config, run_path=args.run_path, stages_to_run=args.stages)

    start = time.perf_counter()
    trainer()
    end = time.perf_counter()

    logger.info(f"Done training in {(end - start) / 60:.4f} minutes")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--rerun", type=str, default=None)
    parser.add_argument("--stages", choices=Trainer.stages.keys(), nargs="+", default=None)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(args)
