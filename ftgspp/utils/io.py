# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import numpy as np

from ftgspp.models.gaussians import Gaussians
from ftgspp.utils.sys import PathLike

PLY_PROPERTY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _parse_ply_header(f):
    if f.readline().strip() != b"ply":
        raise ValueError("Invalid PLY file: missing magic header")

    fmt = None
    vertex_count = None
    vertex_props = []
    current_element = None

    while True:
        line = f.readline()
        if not line:
            raise ValueError("Invalid PLY file: missing end_header")
        line = line.decode("ascii").strip()
        if line == "end_header":
            break

        parts = line.split()
        if not parts or parts[0] == "comment":
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError("PLY vertex list properties are not supported")
            vertex_props.append((parts[2], parts[1]))

    if fmt is None or vertex_count is None:
        raise ValueError("Invalid PLY file: missing format or vertex element")

    return fmt, vertex_count, vertex_props


def _vertex_dtype(vertex_props, byte_order: str = "="):
    return np.dtype(
        [
            (name, np.dtype(byte_order + PLY_PROPERTY_DTYPES[prop]))
            for name, prop in vertex_props
        ]
    )


def read_ply_points(path: PathLike):
    with open(path, "rb") as f:
        fmt, vertex_count, vertex_props = _parse_ply_header(f)
        required_props = {"x", "y", "z", "red", "green", "blue"}
        missing = required_props - {name for name, _prop in vertex_props}
        if missing:
            raise ValueError(f"PLY vertex element is missing properties: {missing}")

        if vertex_count == 0:
            data = {name: np.empty((0,)) for name, _prop in vertex_props}
        elif fmt == "ascii":
            names = [name for name, _prop in vertex_props]
            arr = np.loadtxt(f, max_rows=vertex_count, ndmin=2)
            data = {name: arr[:, idx] for idx, name in enumerate(names)}
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            byte_order = "<" if fmt == "binary_little_endian" else ">"
            vertices = np.fromfile(
                f, dtype=_vertex_dtype(vertex_props, byte_order), count=vertex_count
            )
            data = {name: vertices[name] for name, _prop in vertex_props}
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)

    return xyz, rgb


def write_ply_points(path: PathLike, xyz: np.ndarray, rgb: np.ndarray):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertices = np.empty(len(xyz), dtype)
    vertices["x"] = xyz[..., 0]
    vertices["y"] = xyz[..., 1]
    vertices["z"] = xyz[..., 2]
    vertices["red"] = rgb[..., 0]
    vertices["green"] = rgb[..., 1]
    vertices["blue"] = rgb[..., 2]

    with open(path, "wb") as f:
        f.write(
            "\n".join(
                [
                    "ply",
                    "format binary_little_endian 1.0",
                    f"element vertex {len(vertices)}",
                    "property float x",
                    "property float y",
                    "property float z",
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                    "end_header",
                    "",
                ]
            ).encode("ascii")
        )
        vertices.tofile(f)


def export_npz(model: Gaussians, path: PathLike):
    params = {
        name: param.detach().cpu().numpy() for name, param in model.named_parameters()
    }

    np.savez_compressed(path, allow_pickle=True, **params)
