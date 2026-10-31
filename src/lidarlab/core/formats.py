"""Frames as binary PLY and PCD files, for CloudCompare, PCL, Open3D and similar tools.

Both get x, y, z (float32) and every numeric per-point field of the frame (intensity, echo,
ring, label, …) in its own type; 64-bit integers are written as 32-bit ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .frame import Frame

_PLY_TYPES = {"f4": "float", "f8": "double", "u1": "uchar", "i1": "char", "u2": "ushort",
              "i2": "short", "u4": "uint", "i4": "int"}


def _file_dtype(dtype: np.dtype) -> np.dtype:
    """Little-endian type both formats know: bool -> uint8, 64-bit int -> 32-bit."""
    if dtype.kind == "b":
        return np.dtype("u1")
    if dtype.kind in "iu" and dtype.itemsize == 8:
        return np.dtype(f"<{dtype.kind}4")
    if dtype.kind == "f" and dtype.itemsize == 2:
        return np.dtype("<f4")
    return dtype.newbyteorder("<")


def point_table(frame: Frame) -> np.ndarray:
    """Structured array (packed, little-endian): x, y, z and the numeric per-point fields."""
    columns = [("x", frame.x), ("y", frame.y), ("z", frame.z)]
    for name, values in frame.fields.items():
        if values.ndim == 1 and values.dtype.kind in "biuf" and name not in ("x", "y", "z"):
            columns.append((name, values))
    table = np.empty(len(frame), [(name, _file_dtype(v.dtype)) for name, v in columns])
    for name, values in columns:
        table[name] = values
    return table


def write_ply(frame: Frame, path) -> Path:
    table = point_table(frame)
    header = ["ply", "format binary_little_endian 1.0", "comment written by LidarLab",
              f"element vertex {len(table)}"]
    for name in table.dtype.names:
        header.append(f"property {_PLY_TYPES[table.dtype[name].str[1:]]} {name}")
    header.append("end_header")
    return _write(path, ("\n".join(header) + "\n").encode("ascii"), table)


def write_pcd(frame: Frame, path) -> Path:
    table = point_table(frame)
    names = table.dtype.names
    kinds = {"f": "F", "u": "U", "i": "I", "b": "U"}
    header = [
        "# .PCD v0.7 - Point Cloud Data file format (written by LidarLab)",
        "VERSION 0.7",
        "FIELDS " + " ".join(names),
        "SIZE " + " ".join(str(table.dtype[n].itemsize) for n in names),
        "TYPE " + " ".join(kinds[table.dtype[n].kind] for n in names),
        "COUNT " + " ".join("1" for _ in names),
        f"WIDTH {len(table)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(table)}",
        "DATA binary",
    ]
    return _write(path, ("\n".join(header) + "\n").encode("ascii"), table)


def _write(path, header: bytes, table: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(table.tobytes())
    return path
