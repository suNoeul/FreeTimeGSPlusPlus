# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from PIL import Image
from torch import Tensor

from ftgspp.utils import PathLike


class RoMaConfigLike(Protocol):
    backend: Literal["v1", "v2"]
    model: Literal["indoor", "outdoor"]
    upsample_preds: bool
    symmetric: bool


class RoMaMatcher(Protocol):
    def match_points(
        self,
        img_a: Image.Image,
        img_b: Image.Image,
        num: int,
        debug_path: Optional[PathLike] = None,
    ) -> tuple[Tensor, Tensor]: ...


class RoMaV1Matcher:
    _model: Any

    def __init__(
        self,
        *,
        device: str,
        model: Literal["indoor", "outdoor"],
        upsample_preds: bool,
        symmetric: bool,
    ):
        import romatch

        factories = {"indoor": romatch.roma_indoor, "outdoor": romatch.roma_outdoor}
        self._model = factories[model](device)
        self._model.symmetric = symmetric
        self._model.upsample_preds = upsample_preds
        self._model.sample_mode = "threshold"

    def match_points(
        self,
        img_a: Image.Image,
        img_b: Image.Image,
        num: int,
        debug_path: Optional[PathLike] = None,
    ) -> tuple[Tensor, Tensor]:
        warp, certainty = self._model.match(img_a, img_b)
        sampled, _ = self._model.sample(warp, certainty, num=num)

        if debug_path is not None:
            self._model.visualize_warp(
                warp,
                certainty,
                img_a,
                img_b,
                symmetric=self._model.symmetric,
                save_path=Path(debug_path),
            )

        pts_a, pts_b = self._model.to_pixel_coordinates(
            sampled, img_a.height, img_a.width, img_b.height, img_b.width
        )
        return pts_a.cpu(), pts_b.cpu()


class RoMaV2Matcher:
    _model: Any

    def __init__(
        self,
        *,
        device: str,
        model: Literal["indoor", "outdoor"],
        upsample_preds: bool,
        symmetric: bool,
    ):
        try:
            from romav2 import RoMaV2
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "RoMa v2 backend requires `romav2`. Install with `uv pip install romav2` "
                "or switch `init.roma.backend` to `v1`."
            ) from e

        self._model = RoMaV2()
        if hasattr(self._model, "to"):
            self._model = self._model.to(device)
        if hasattr(self._model, "eval"):
            self._model.eval()

        if hasattr(self._model, "apply_setting"):
            try:
                self._model.apply_setting(model)
            except Exception:
                pass

        # Different RoMa versions expose different knobs; set when available.
        if hasattr(self._model, "symmetric"):
            self._model.symmetric = symmetric
        if hasattr(self._model, "upsample_preds"):
            self._model.upsample_preds = upsample_preds
        if hasattr(self._model, "sample_mode"):
            self._model.sample_mode = "threshold"

    @staticmethod
    def _extract_sampled_matches(sampled: Any) -> Any:
        if isinstance(sampled, dict):
            for key in ("matches", "coords", "correspondences"):
                if key in sampled:
                    return sampled[key]
            raise ValueError("RoMa v2 sample() output dict has no known match key")

        if isinstance(sampled, (list, tuple)):
            if len(sampled) == 0:
                raise ValueError("RoMa v2 sample() returned an empty tuple/list")
            return sampled[0]

        return sampled

    def match_points(
        self,
        img_a: Image.Image,
        img_b: Image.Image,
        num: int,
        debug_path: Optional[PathLike] = None,
    ) -> tuple[Tensor, Tensor]:
        del debug_path

        preds = self._model.match(img_a, img_b)
        try:
            sampled = self._model.sample(preds, num=num)
        except TypeError:
            sampled = self._model.sample(preds, num)

        matches = self._extract_sampled_matches(sampled)

        try:
            pts_a, pts_b = self._model.to_pixel_coordinates(
                matches, img_a.height, img_a.width, img_b.height, img_b.width
            )
        except TypeError:
            pts_a, pts_b = self._model.to_pixel_coordinates(
                matches, img_a.height, img_a.width
            )

        return pts_a.cpu(), pts_b.cpu()


def create_roma_matcher(
    config: RoMaConfigLike,
    *,
    device: str,
) -> RoMaMatcher:
    match config.backend:
        case "v1":
            return RoMaV1Matcher(
                device=device,
                model=config.model,
                upsample_preds=config.upsample_preds,
                symmetric=config.symmetric,
            )
        case "v2":
            return RoMaV2Matcher(
                device=device,
                model=config.model,
                upsample_preds=config.upsample_preds,
                symmetric=config.symmetric,
            )

    raise ValueError(f"Unsupported RoMa backend: {config.backend}")
