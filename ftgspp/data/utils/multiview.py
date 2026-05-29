# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional, Self, Sequence, cast

import cv2
import numpy as np
import pycolmap
import tensordict
import torch
from tensordict import NonTensorData, TensorClass
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import Resize
from tqdm import tqdm

from ftgspp.data.utils.images import scan_images, to_image_name
from ftgspp.utils import Interval, PathLike, num_workers

tensordict.set_list_to_stack(True).set()


@dataclass
class _MultiViewImages(Dataset):
    path: Path
    suffix: str
    frames: Sequence[int]
    cameras: Sequence[int]
    transform: Callable[[Tensor], Tensor] = lambda x: x
    undistort_maps: Optional[Sequence[tuple[np.ndarray, np.ndarray]]] = None
    undistort_roi: Optional[tuple[int, int, int, int]] = None
    undistort_size: Optional[tuple[int, int]] = None

    def __len__(self):
        return len(self.frames) * len(self.cameras)

    def __getitem__(self, idx: int):
        frame = self.frames[idx // len(self.cameras)]
        camera = self.cameras[idx % len(self.cameras)]
        name = to_image_name(frame=frame, camera=camera, suffix=self.suffix)

        img_path = str(self.path / name)
        if self.undistort_maps is None:
            img = decode_image(img_path, mode=ImageReadMode.RGB)
            return self.transform(img)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        map1, map2 = self.undistort_maps[camera]
        img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        x, y, w, h = (
            self.undistort_roi
            if self.undistort_roi is not None
            else (0, 0, img.shape[1], img.shape[0])
        )
        img = img[y : y + h, x : x + w]
        if (
            self.undistort_size is not None
            and (img.shape[1], img.shape[0]) != self.undistort_size
        ):
            img = cv2.resize(img, self.undistort_size, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img


class MultiViewData(TensorClass, nocast=True):  # type: ignore
    """
    Represents a multiview video dataset with camera parameters. Batched as
    `(num_frames, num_cameras)`.
    """

    rgb: Tensor  # (num_frames, num_cameras, height, width, color)
    w2c: Tensor  # (num_frames, num_cameras, 4, 4)
    intrinsic: Tensor  # (num_frames, num_cameras, 3, 3)
    time: Tensor  # (num_frames, num_cameras, 1)

    frame: list[list[int]] | list[int] | int
    camera: list[list[int]] | list[int] | int
    path: list[list[Path]] | list[Path] | Path  # (num_frames, num_cameras)

    @classmethod
    def new(
        cls,
        image_path: PathLike,
        out_path: PathLike,
        colmap_path: PathLike,
        frame_range: Interval,
        fps: Optional[float] = None,
        scale: float = 1,
    ) -> Self:
        image_path = Path(image_path)
        if fps is None:
            fps = 30
            warnings.warn(f"no fps provided, falling back to {fps}fps")

        frames, cameras, suffix = scan_images(image_path, frame_range)

        undistort_maps = None
        undistort_roi = None
        undistort_size = None
        undistort_intrinsics = None

        from ftgspp.data.sfm import load_camera_metadata

        camera_meta = load_camera_metadata(colmap_path)
        if camera_meta is not None:
            if abs(camera_meta.scale - scale) > 1e-9:
                raise ValueError(
                    f"camera metadata scale mismatch: {camera_meta.scale} != {scale}"
                )

            cam_ids = sorted(camera_meta.names, key=int)
            expected_ids = list(cameras)
            if [int(cam_id) for cam_id in cam_ids] != expected_ids:
                raise ValueError(
                    f"camera metadata IDs do not match extracted cameras: {cam_ids} != {expected_ids}"
                )

            undistort_maps = []
            for cam_id in cam_ids:
                k = np.asarray(camera_meta.ks[cam_id], dtype=np.float64)
                d = np.asarray(camera_meta.ds[cam_id], dtype=np.float64).reshape(-1, 1)
                new_k = np.asarray(camera_meta.new_ks[cam_id], dtype=np.float64)
                h, w = camera_meta.shapes[cam_id]
                map1, map2 = cv2.initUndistortRectifyMap(
                    k, d, np.eye(3, dtype=np.float64), new_k, (w, h), cv2.CV_32FC1
                )
                undistort_maps.append((map1, map2))

            undistort_roi = camera_meta.roi
            undistort_size = camera_meta.undistort_size
            undistort_intrinsics = torch.tensor(
                np.stack(
                    [camera_meta.undistort_intrinsics[cam_id] for cam_id in cam_ids],
                    axis=0,
                )
            )

            height = undistort_size[1]
            width = undistort_size[0]
            transform = None
        else:
            sample = image_path / (to_image_name(frames[0], cameras[0], suffix))
            img_sample = decode_image(str(sample), mode=ImageReadMode.RGB)

            _, original_height, original_width = img_sample.shape
            transform = Resize(
                (int(scale * original_height), int(scale * original_width))
            )

            _, height, width = transform(img_sample).shape

        w2c, intrinsic = read_sfm(
            colmap_path,
            image_names=[to_image_name(frames[0], c, suffix=suffix) for c in cameras],
        )
        w2c = torch.tensor(w2c)
        if undistort_intrinsics is None:
            intrinsic = torch.tensor(intrinsic) * scale
            intrinsic[..., 2, 2] = 1
        else:
            intrinsic = undistort_intrinsics

        self = cls(
            rgb=torch.empty((height, width, 3), dtype=torch.uint8),
            w2c=torch.empty((4, 4), dtype=torch.float32),
            intrinsic=torch.empty((3, 3), dtype=torch.float32),
            time=torch.empty((1,), dtype=torch.float32),
            path=Path(""),
            frame=-1,
            camera=-1,
        )

        self: Self = self.expand(len(frames), len(cameras))

        self.path = [
            [image_path / to_image_name(f, c, suffix) for c in cameras] for f in frames
        ]
        self.frame = cast(
            list[list[int]],
            tensordict.stack(
                [tensordict.stack([NonTensorData(f) for _ in cameras]) for f in frames]
            ),
        )
        self.camera = cast(
            list[list[int]],
            tensordict.stack(
                [tensordict.stack([NonTensorData(c) for c in cameras]) for _ in frames]
            ),
        )

        self: Self = self.memmap_like(str(out_path))

        images = _MultiViewImages(
            path=image_path,
            suffix=suffix,
            frames=frames,
            cameras=cameras,
            transform=transform if transform is not None else (lambda x: x),
            undistort_maps=undistort_maps,
            undistort_roi=undistort_roi,
            undistort_size=undistort_size,
        )
        loader = DataLoader(images, batch_size=None, num_workers=num_workers())

        for i, image in tqdm(enumerate(loader), total=len(images)):
            frame = i // len(cameras)
            camera = i % len(cameras)
            self.rgb[frame, camera] = image.permute(1, 2, 0)

        for i, frame in enumerate(frames):
            self.w2c[i] = w2c
            self.intrinsic[i] = intrinsic
            self.time[i] = frame / fps

        return self

    @cached_property
    def num_frames(self) -> int:
        if self.rgb.dim() == 5:
            return self.rgb.size(-5)
        else:
            raise ValueError("unknown num_frames")

    @cached_property
    def num_cameras(self) -> int:
        if self.rgb.dim() == 5:
            return self.rgb.size(-4)
        else:
            raise ValueError("unknown num_cameras")

    @cached_property
    def fps(self) -> float:
        if self.rgb.dim() == 5:
            return 1 / float(self.time[1, 0] - self.time[0, 0])
        else:
            raise ValueError("unknown fps")

    @cached_property
    def duration(self) -> float:
        if self.rgb.dim() == 5:
            dt = self.time[1, 0] - self.time[0, 0]
            return float(self.time[-1, 0] - self.time[0, 0] + dt)
        else:
            raise ValueError("unknown duration")

    @cached_property
    def height(self) -> int:
        return self.rgb.size(-3)

    @cached_property
    def width(self) -> int:
        return self.rgb.size(-2)

    def at(
        self,
        frame: Optional[slice] = None,
        camera: Optional[slice | Sequence[int]] = None,
    ) -> Self:
        assert frame is None or 0 <= frame.start <= frame.stop
        assert frame is None or frame.stop - frame.start <= self.num_frames
        if isinstance(camera, slice):
            assert 0 <= camera.start <= camera.stop
            assert camera.stop - camera.start <= self.num_cameras

        if frame is not None:
            zero_frame = int(self.flatten()[0].frame)  # type: ignore
            frame_index = slice(frame.start - zero_frame, frame.stop - zero_frame)
        else:
            frame_index = slice(None)

        if camera is not None:
            zero_camera = int(self.flatten()[0].camera)  # type: ignore
            if isinstance(camera, slice):
                camera_index = slice(
                    camera.start - zero_camera, camera.stop - zero_camera
                )
            else:
                camera_index = [c - zero_camera for c in camera]
        else:
            camera_index = slice(None)

        return self[frame_index, camera_index]  # type: ignore


def read_sfm(recon_path: PathLike, image_names: Sequence[str]):
    w2c = np.zeros((len(image_names), 4, 4), dtype=np.float64)
    intrinsic = np.zeros((len(image_names), 3, 3), dtype=np.float64)

    recon = pycolmap.Reconstruction(str(recon_path))

    for i, name in enumerate(image_names):
        image = recon.find_image_with_name(name)
        camera = image.camera
        assert image is not None and camera is not None

        w2c[i, :3, :3] = image.cam_from_world().rotation.matrix()
        w2c[i, :3, 3] = image.cam_from_world().translation
        w2c[i, 3, :3] = 0
        w2c[i, 3, 3] = 1

        intrinsic[i] = camera.calibration_matrix().copy()

    return w2c, intrinsic
