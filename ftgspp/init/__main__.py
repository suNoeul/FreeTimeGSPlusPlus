# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import argparse
from pathlib import Path

from gsplat import export_splats

from ftgspp.data.utils import MultiViewData
from ftgspp.init import init
from ftgspp.utils import Config


def main(args: argparse.Namespace):
    config = Config.load(args.config)
    run_path: Path = args.run_path

    dataset: MultiViewData = MultiViewData.load_memmap(config.data.memmap_path)
    dataset = dataset.at(config.data.frames.into_slice())

    gs = init(
        dataset=dataset,
        num_gaussians=config.init.num_gaussians,
        keyframe_stride=config.init.keyframe_stride,
        points_path=config.init.points_path,
        sh_degree=config.init.sh_degree,
        opacity=config.init.opacity,
        scale=config.init.scale,
        duration=config.init.duration,
        num_velocity_nns=config.init.num_velocity_nns,
        temporal_motion_adapted=config.init.temporal_motion_adapted,
        temporal_flow_path=config.init.temporal_flow_path,
        temporal_covis_thresh=config.init.temporal_covis_thresh,
        max_duration=config.model.max_duration,
    )

    run_path.mkdir(exist_ok=True, parents=True)
    gs.save(run_path / "init.pt")
    export_splats(
        means=gs.means,
        scales=gs.scales,
        quats=gs.quats,
        opacities=gs.opacities.squeeze(-1),
        sh0=gs.sh_0,
        shN=gs.sh_n,
        format="ply",
        save_to=str(run_path / "init.ply"),
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("run_path", type=Path)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(args)
