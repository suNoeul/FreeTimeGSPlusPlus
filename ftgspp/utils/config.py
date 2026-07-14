# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab


from pathlib import Path
from typing import Literal, Self

import msgspec

from ftgspp.utils.sys import PathLike


class ConfigBase(
    msgspec.Struct,
    forbid_unknown_fields=True,
): ...


class Interval(ConfigBase):
    start: int  # inclusive
    stop: int  # exclusive

    def into_range(self) -> range:
        return range(self.start, self.stop)

    def into_slice(self) -> slice:
        return slice(self.start, self.stop)


CameraSelection = Interval | list[int]


def camera_indexer(cameras: CameraSelection) -> slice | list[int]:
    if isinstance(cameras, Interval):
        return cameras.into_slice()
    return list(cameras)


class ModelConfig(ConfigBase):
    velocity_model: Literal["explicit", "field"]
    max_duration: float
    marginal_gating: bool


class DataConfig(ConfigBase):
    fps: float
    video_path: Path
    calibration_path: Path
    extracted_path: Path
    memmap_path: Path
    colmap_path: Path

    scale: float
    eval_cameras: CameraSelection
    train_cameras: CameraSelection
    frames: Interval


class InitConfig(ConfigBase):
    points_path: Path
    keyframe_stride: int
    num_points_per_frame: int
    num_gaussians: int
    sh_degree: int
    opacity: float
    scale: float
    duration: float
    num_velocity_nns: int

    class RoMaConfig(ConfigBase):
        model: Literal["indoor", "outdoor"]
        num_nearest_cameras: int
        upsample_preds: bool
        symmetric: bool
        geometric_verification: bool
        backend: Literal["v1", "v2"] = "v1"

    roma: RoMaConfig
    temporal_motion_adapted: bool = False
    temporal_flow_path: Path = Path("_flow")
    temporal_covis_thresh: float = 0.3


class TrainConfig(ConfigBase):
    iterations: int
    batch_size: int
    lrs: dict[str, float]
    lr_schedules: dict[str, float]

    class RelocationConfig(ConfigBase):
        start: int
        stop: int
        every: int
        opacity_threshold: float
        mode: Literal["partial_copy", "exact_copy", "3d_mcmc"] = "3d_mcmc"
        score_mode: Literal["default", "gate_included"] = "default"
        score_mode_start: int = 3000

    relocation: RelocationConfig
    lpips_loss: bool

    color_correction: bool
    color_correction_start: int
    color_corrector_weight: float

    hard_separation: bool
    seed: int = 0


def deep_update(base: dict, new: dict):
    for k, v in new.items():
        if isinstance(v, dict):
            base[k] = deep_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


class Config(ConfigBase):
    model: ModelConfig
    data: DataConfig
    init: InitConfig
    train: TrainConfig

    @classmethod
    def load(cls, path: PathLike) -> Self:
        path = Path(path)

        def _dec_hook(type: type, obj):
            if type is Path:
                return Path(obj).expanduser().resolve()
            return obj

        obj = msgspec.toml.decode(path.read_bytes(), type=dict)
        if "extends" in obj:
            base_path = path.parent / (obj["extends"])
            base_obj = msgspec.toml.decode(base_path.read_bytes(), type=dict)
            obj = deep_update(base_obj, obj)
            del obj["extends"]

        config = msgspec.convert(
            obj,
            type=cls,
            dec_hook=_dec_hook,
        )

        return config

    def save(self, path: PathLike):
        def _enc_hook(obj):
            if isinstance(obj, Path):
                return str(obj.expanduser().resolve())
            return obj

        def _drop_none(obj):
            if isinstance(obj, dict):
                return {k: _drop_none(v) for k, v in obj.items() if v is not None}
            if isinstance(obj, list):
                return [_drop_none(v) for v in obj]
            return obj

        with open(path, "wb") as f:
            config = msgspec.to_builtins(self, enc_hook=_enc_hook)
            config = _drop_none(config)
            config = msgspec.toml.encode(config)
            f.write(config)
