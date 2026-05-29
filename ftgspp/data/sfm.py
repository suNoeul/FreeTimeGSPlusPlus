# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pycolmap

from ftgspp.data.utils.images import scan_images, to_image_name
from ftgspp.utils import Config, PathLike

CAMERA_METADATA_FILENAME = "camera_metadata.npz"


def parse_poses_bounds_npy(
    path: PathLike, image_names: list[str]
) -> pycolmap.Reconstruction:
    arr: np.ndarray = np.load(path)
    cam = arr[:, :-2].reshape((-1, 3, 5))
    _bounds = arr[:, -2:]

    c2ws = cam[..., :4]
    hwfs = cam[..., 4]
    c2ws = np.stack([c2ws[..., 1], c2ws[..., 0], -c2ws[..., 2], c2ws[..., 3]], axis=1)

    bottom = np.array([0, 0, 0, 1]).reshape((1, 4, 1)).repeat(len(c2ws), 0)
    c2ws = np.concatenate([c2ws, bottom], axis=2)
    w2cs = np.linalg.inv(c2ws).mT

    recon = pycolmap.Reconstruction()

    ids = range(1, len(image_names) + 1)
    for id, name, w2c, (h, w, f) in zip(ids, image_names, w2cs, hwfs):
        pose = pycolmap.Rigid3d(w2c[..., :3, :4])

        cam = pycolmap.Camera.create(
            camera_id=id,
            model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
            focal_length=f,
            width=int(w),
            height=int(h),
        )
        recon.add_camera(cam)
        cam = recon.camera(cam.camera_id)

        rig = pycolmap.Rig(rig_id=id)
        rig.add_ref_sensor(cam.sensor_id)
        recon.add_rig(rig)
        rig = recon.rig(rig.rig_id)

        frame = pycolmap.Frame(frame_id=id, rig_id=rig.rig_id)
        img = pycolmap.Image(
            name=name,
            image_id=id,
            camera_id=cam.camera_id,
            frame_id=frame.frame_id,
        )

        recon.add_frame(frame)
        frame = recon.frame(frame.frame_id)
        recon.add_image(img)
        img = recon.image(img.image_id)

        frame.set_cam_from_world(cam.camera_id, pose)
        frame.add_data_id(img.data_id)
        recon.register_image(id)

    print(recon.summary())

    return recon


# See https://github.com/zju3dv/EasyVolcap/blob/4cb3c000a31b8764834c79792b355f110d947e75/easyvolcap/utils/camera_utils.py#L100
# for intrinsic/extrinsic convention used for OpenCV handling.


@dataclass
class OpenCVIntrinsics:
    names: list[str]
    ks: dict[str, np.ndarray]  # intrinsic
    ds: dict[str, np.ndarray]  # distortion
    shapes: dict[str, tuple[float, float]]

    @classmethod
    def read(cls, path: PathLike):
        fstore = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        names_node = fstore.getNode("names")
        names = [names_node.at(i).string() for i in range(names_node.size())]

        self = cls(names=names, ks={}, ds={}, shapes={})
        for name in names:
            self.ks[name] = np.array(
                fstore.getNode(f"K_{name}").mat(), dtype=np.float64
            )
            self.ds[name] = np.array(
                fstore.getNode(f"D_{name}").mat(), dtype=np.float64
            ).reshape(-1)
            self.shapes[name] = (
                fstore.getNode(f"H_{name}").real(),
                fstore.getNode(f"W_{name}").real(),
            )

        return self


@dataclass
class OpenCVExtrinsics:
    names: list[str]
    rots: dict[str, np.ndarray]  # intrinsic
    ts: dict[str, np.ndarray]  # distortion

    @classmethod
    def read(cls, path: PathLike):
        fstore = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        names_node = fstore.getNode("names")
        names = [names_node.at(i).string() for i in range(names_node.size())]

        self = cls(names=names, rots={}, ts={})
        for name in names:
            self.rots[name] = np.array(
                fstore.getNode(f"Rot_{name}").mat(), dtype=np.float64
            )
            self.ts[name] = np.array(
                fstore.getNode(f"T_{name}").mat(), dtype=np.float64
            ).reshape(3, 1)

        return self


@dataclass
class OpenCVUndistortMetadata:
    names: list[str]
    roi: tuple[int, int, int, int]
    undistort_size: tuple[int, int]
    scale: float
    ks: dict[str, np.ndarray]
    ds: dict[str, np.ndarray]
    new_ks: dict[str, np.ndarray]
    shapes: dict[str, tuple[int, int]]
    undistort_intrinsics: dict[str, np.ndarray]

    @classmethod
    def from_intrinsics(cls, intri: OpenCVIntrinsics, scale: float):
        names = sorted(intri.names, key=int)

        new_ks: dict[str, np.ndarray] = {}
        rois: list[tuple[int, int, int, int]] = []
        for name in names:
            k = intri.ks[name]
            d = intri.ds[name].reshape(-1, 1)
            h, w = intri.shapes[name]
            new_k, roi = cv2.getOptimalNewCameraMatrix(
                k, d, (int(w), int(h)), 0, (int(w), int(h))
            )
            new_ks[name] = new_k
            rois.append(tuple(int(v) for v in roi))

        x0 = max(r[0] for r in rois)
        y0 = max(r[1] for r in rois)
        x1 = min(r[0] + r[2] for r in rois)
        y1 = min(r[1] + r[3] for r in rois)
        roi = (x0, y0, x1 - x0, y1 - y0)

        out_w = int(scale * roi[2])
        out_h = int(scale * roi[3])
        undistort_size = (out_w, out_h)

        undistort_intrinsics: dict[str, np.ndarray] = {}
        for name in names:
            k_adj = new_ks[name].copy()
            k_adj[0, 2] -= roi[0]
            k_adj[1, 2] -= roi[1]
            k_adj[:2] *= scale
            k_adj[2, 2] = 1
            undistort_intrinsics[name] = k_adj

        shapes = {
            name: (int(intri.shapes[name][0]), int(intri.shapes[name][1]))
            for name in names
        }

        return cls(
            names=names,
            roi=roi,
            undistort_size=undistort_size,
            scale=scale,
            ks={name: intri.ks[name].copy() for name in names},
            ds={name: intri.ds[name].copy() for name in names},
            new_ks=new_ks,
            shapes=shapes,
            undistort_intrinsics=undistort_intrinsics,
        )

    def write(self, path: PathLike):
        path = Path(path)
        arrays: dict[str, np.ndarray] = {
            "names": np.asarray(self.names),
            "roi": np.asarray(self.roi, dtype=np.int32),
            "undistort_size": np.asarray(self.undistort_size, dtype=np.int32),
            "scale": np.asarray(self.scale, dtype=np.float64),
        }
        for name in self.names:
            arrays[f"k_{name}"] = self.ks[name]
            arrays[f"d_{name}"] = self.ds[name]
            arrays[f"new_k_{name}"] = self.new_ks[name]
            arrays[f"shape_{name}"] = np.asarray(self.shapes[name], dtype=np.int32)
            arrays[f"undistort_k_{name}"] = self.undistort_intrinsics[name]
        np.savez(path, **arrays)

    @classmethod
    def read(cls, path: PathLike):
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        names = [str(name) for name in data["names"].tolist()]
        ks = {name: data[f"k_{name}"] for name in names}
        ds = {name: data[f"d_{name}"] for name in names}
        new_ks = {name: data[f"new_k_{name}"] for name in names}
        shapes = {
            name: tuple(int(v) for v in data[f"shape_{name}"].tolist()) for name in names
        }
        undistort_intrinsics = {name: data[f"undistort_k_{name}"] for name in names}

        return cls(
            names=names,
            roi=tuple(int(v) for v in data["roi"].tolist()),
            undistort_size=tuple(int(v) for v in data["undistort_size"].tolist()),
            scale=float(data["scale"]),
            ks=ks,
            ds=ds,
            new_ks=new_ks,
            shapes=shapes,
            undistort_intrinsics=undistort_intrinsics,
        )


def camera_metadata_path(colmap_path: PathLike) -> Path:
    return Path(colmap_path) / CAMERA_METADATA_FILENAME


def load_camera_metadata(colmap_path: PathLike) -> Optional[OpenCVUndistortMetadata]:
    path = camera_metadata_path(colmap_path)
    if not path.exists():
        return None
    return OpenCVUndistortMetadata.read(path)


def parse_opencv_calibration(
    calib_path: PathLike,
    image_names: list[str],
    scale: float,
) -> tuple[pycolmap.Reconstruction, OpenCVUndistortMetadata]:
    calib_path = Path(calib_path)
    intri_path = calib_path / "intri.yml"
    extri_path = calib_path / "extri.yml"

    intri = OpenCVIntrinsics.read(intri_path)
    extri = OpenCVExtrinsics.read(extri_path)
    undistort_meta = OpenCVUndistortMetadata.from_intrinsics(intri, scale)

    if len(intri.names) != len(image_names):
        raise ValueError("image count does not match intrinsic calibration")
    if len(extri.names) != len(image_names):
        raise ValueError("image count does not match extrinsic calibration")
    if sorted(intri.names) != sorted(extri.names):
        raise ValueError("intrinsic and extrinsic camera names do not match")

    cam_names = sorted(intri.names)
    recon = pycolmap.Reconstruction()

    for image_id, (image_name, cam_name) in enumerate(
        zip(image_names, cam_names), start=1
    ):
        h, w = intri.shapes[cam_name]
        k = intri.ks[cam_name]
        d = intri.ds[cam_name]
        rot = extri.rots[cam_name]
        trans = extri.ts[cam_name]

        fx = float(k[0, 0])
        fy = float(k[1, 1])
        cx = float(k[0, 2])
        cy = float(k[1, 2])

        cam = pycolmap.Camera.create(
            camera_id=image_id,
            model=pycolmap.CameraModelId.FULL_OPENCV,
            focal_length=fx,
            width=int(w),
            height=int(h),
        )
        cam.params = [fx, fy, cx, cy, *d.tolist(), 0.0, 0.0, 0.0]
        recon.add_camera(cam)
        cam = recon.camera(cam.camera_id)

        rig = pycolmap.Rig(rig_id=image_id)
        rig.add_ref_sensor(cam.sensor_id)
        recon.add_rig(rig)
        rig = recon.rig(rig.rig_id)

        frame = pycolmap.Frame(frame_id=image_id, rig_id=rig.rig_id)
        img = pycolmap.Image(
            name=image_name,
            image_id=image_id,
            camera_id=cam.camera_id,
            frame_id=frame.frame_id,
        )

        recon.add_frame(frame)
        frame = recon.frame(frame.frame_id)
        recon.add_image(img)
        img = recon.image(img.image_id)

        rotation = pycolmap.Rotation3d(rot)
        pose = pycolmap.Rigid3d(rotation, trans)
        frame.set_cam_from_world(cam.camera_id, pose)
        frame.add_data_id(img.data_id)
        recon.register_image(image_id)

    print(recon.summary())

    return recon, undistort_meta


def colmap_sfm(image_path: PathLike, image_names: list[str]) -> pycolmap.Reconstruction:
    with tempfile.TemporaryDirectory() as dir:
        dir = Path(dir)
        db_path = dir / "colmap.db"
        colmap_path = dir / "colmap"

        pycolmap.extract_features(
            str(db_path),
            str(image_path),
            camera_mode=pycolmap.CameraMode.PER_IMAGE,
            camera_model=pycolmap.CameraModelId.SIMPLE_PINHOLE.name,
            image_names=image_names,
        )

        pycolmap.match_exhaustive(str(db_path))

        recon = pycolmap.incremental_mapping(
            str(db_path), str(image_path), str(colmap_path)
        )[0]

    return recon


def main(args: argparse.Namespace):
    config = Config.load(args.config)

    frames, cameras, suffix = scan_images(
        config.data.extracted_path, config.data.frames
    )
    image_names = [to_image_name(frames[0], c, suffix=suffix) for c in cameras]

    undistort_meta: Optional[OpenCVUndistortMetadata] = None
    if config.data.calibration_path.suffix == ".npy":
        recon = parse_poses_bounds_npy(config.data.calibration_path, image_names)
    elif config.data.calibration_path.name == "optimized":
        recon, undistort_meta = parse_opencv_calibration(
            config.data.calibration_path,
            image_names,
            scale=config.data.scale,
        )
    else:
        raise ValueError(f"unsupported calibration file {config.data.calibration_path}")

    config.data.colmap_path.mkdir(exist_ok=True, parents=True)
    recon.write_text(str(config.data.colmap_path))
    meta_path = camera_metadata_path(config.data.colmap_path)
    if undistort_meta is not None:
        undistort_meta.write(meta_path)
    elif meta_path.exists():
        meta_path.unlink()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(args)
