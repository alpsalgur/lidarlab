"""Reading and writing frames as ``.npz`` files.

Layouts, detected per file:

- echo-axis: point keys have shape (E, P): row = echo, column = pulse, NaN = no return.
  Keys of shape (P,) belong to the pulse (e.g. ``ring``) and are copied to its echoes.
- flat: valid points only, shape (M,); the echo number is in ``echo_number`` (or another
  name from ``ECHO_KEYS``) and becomes the ``echo`` field.
- points: one (M, 3+) array called ``points`` (x, y, z[, intensity]), read as flat.

0-d keys (``roll``, ``pitch``, ``yaw``, ``timestamp``) become ``frame.meta`` entries.
``write_frame`` restores the source layout; in echo-axis files removed points become NaN.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from .frame import Frame

ECHO_KEYS = ("echo_number", "echo", "echo_id", "return_number")
FIELD_ORDER = ("intensity", "reflectivity", "echo", "ring")


def read_frame(path, frame_no: int | None = None) -> Frame:
    """Read one frame file; ``frame_no`` is stored as ``meta["frame"]``."""
    with np.load(path, allow_pickle=False) as f:
        arrays = {key: f[key] for key in f.files}
    frame = frame_from_arrays(arrays)
    frame.meta["path"] = str(path)
    if frame_no is not None:
        frame.meta["frame"] = frame_no
    return frame


def frame_from_arrays(arrays) -> Frame:
    """Build a Frame from the arrays of one file, detecting the layout."""
    arrays = dict(arrays)
    if "x" not in arrays and "points" in arrays:
        pts = arrays.pop("points")
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"'points' must have shape (M, 3+), got {pts.shape}")
        arrays.update(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2])
        if pts.shape[1] > 3:
            arrays.setdefault("intensity", pts[:, 3])
    missing = [k for k in ("x", "y", "z") if k not in arrays]
    if missing:
        raise KeyError(f"no point coordinates {missing}; keys in the file: {sorted(arrays)}")
    x, y, z = (arrays.pop(k) for k in ("x", "y", "z"))
    if not x.shape == y.shape == z.shape:
        raise ValueError(f"x, y, z shapes differ: {x.shape}, {y.shape}, {z.shape}")
    scalar_keys = [k for k, v in arrays.items() if v.ndim == 0]
    meta = {k: arrays.pop(k)[()] for k in scalar_keys}
    meta["scalar_keys"] = scalar_keys
    if x.ndim == 2:
        return _from_echo_axis(x, y, z, arrays, meta)
    if x.ndim == 1:
        return _from_flat(x, y, z, arrays, meta)
    raise ValueError(f"x has shape {x.shape}; expected (M,) or (echoes, pulses)")


def _index_dtype(size: int):
    return np.int32 if size < 2**31 else np.int64


def _ordered(fields: dict) -> dict:
    first = [k for k in FIELD_ORDER if k in fields]
    return {k: fields[k] for k in [*first, *(k for k in fields if k not in first)]}


def _valid(x, y, z) -> np.ndarray:
    valid = np.isfinite(x)
    valid &= np.isfinite(y)
    valid &= np.isfinite(z)
    return valid


def _from_echo_axis(x, y, z, arrays, meta) -> Frame:
    n_echo, n_pulse = x.shape
    idx = np.flatnonzero(_valid(x, y, z))
    xyz = np.empty((idx.size, 3), np.float32)
    for col, a in enumerate((x, y, z)):
        xyz[:, col] = np.take(a.reshape(-1), idx)
    fields = {"echo": (idx // n_pulse).astype(np.min_scalar_type(n_echo - 1))}
    pulse_arrays, extra = {}, {}
    pulse = None
    for key, a in arrays.items():
        if a.shape == x.shape:
            fields[key] = np.take(a.reshape(-1), idx)
        elif a.shape == (n_pulse,):
            if pulse is None:
                pulse = idx % n_pulse
            fields[key] = a[pulse]
            pulse_arrays[key] = a
        else:
            extra[key] = a
    meta.update(layout="echo-axis", shape=(n_echo, n_pulse), pulse_arrays=pulse_arrays,
                extra=extra)
    return Frame(xyz, _ordered(fields), meta, idx.astype(_index_dtype(x.size)))


def _from_flat(x, y, z, arrays, meta) -> Frame:
    n = len(x)
    valid = _valid(x, y, z)
    idx = None if valid.all() else np.flatnonzero(valid)

    def take(a):
        return a if idx is None else a[idx]

    xyz = np.empty((n if idx is None else idx.size, 3), np.float32)
    for col, a in enumerate((x, y, z)):
        xyz[:, col] = take(a)
    echo_key = next((k for k in ECHO_KEYS if k in arrays and arrays[k].shape == (n,)), None)
    fields, extra = {}, {}
    for key, a in arrays.items():
        if a.shape[:1] == (n,):
            fields["echo" if key == echo_key else key] = take(a)
        else:
            extra[key] = a
    if echo_key is not None:
        meta["echo_dtype"] = fields["echo"].dtype.str
        fields["echo"] = fields["echo"].astype(np.uint8)
    meta.update(layout="flat", echo_key=echo_key, extra=extra)
    index = np.arange(n) if idx is None else idx
    return Frame(xyz, _ordered(fields), meta, index.astype(_index_dtype(n)))


def write_frame(frame: Frame, path, layout: str | None = None, compress: bool = True) -> Path:
    """Write ``frame`` as .npz in ``layout`` (default: the layout it was read from, else flat)."""
    layout = layout or frame.meta.get("layout", "flat")
    if layout == "echo-axis":
        arrays = _to_echo_axis(frame)
    elif layout == "flat":
        arrays = _to_flat(frame)
    else:
        raise ValueError(f"unknown layout {layout!r}; use 'echo-axis' or 'flat'")
    for key in frame.meta.get("scalar_keys", ()):
        if key in frame.meta:
            arrays[key] = np.asarray(frame.meta[key])
    for key, a in frame.meta.get("extra", {}).items():
        arrays.setdefault(key, a)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        save_compressed(path, arrays)
    else:
        np.savez(path, **arrays)
    return path


def save_compressed(path, arrays: dict, level: int = 1) -> None:
    """Like ``np.savez_compressed`` (any reader loads it), at zlib level 1 instead of 6:
    a third less time for files ~3 % larger."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
        for key, a in arrays.items():
            with zf.open(f"{key}.npy", "w", force_zip64=True) as out:
                np.lib.format.write_array(out, np.asanyarray(a), allow_pickle=False)


def _to_echo_axis(frame: Frame) -> dict:
    shape = frame.meta.get("shape")
    if shape is None:
        raise ValueError("echo-axis output needs meta['shape'] = (echoes, pulses); "
                         "write this frame with layout='flat'")
    size = shape[0] * shape[1]
    idx = np.asarray(frame.index, np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= size):
        raise ValueError(f"point index outside the {shape[0]} x {shape[1]} echo grid")
    pulse_arrays = frame.meta.get("pulse_arrays", {})

    def scatter(values):
        fill = np.nan if np.issubdtype(values.dtype, np.floating) else 0
        out = np.full(size, fill, values.dtype)
        out[idx] = values
        return out.reshape(shape)

    arrays = {k: scatter(frame.xyz[:, col]) for col, k in enumerate("xyz")}
    for key, values in frame.fields.items():
        # the echo is the row; pulse arrays are written unchanged below
        if key != "echo" and key not in pulse_arrays and values.ndim == 1:
            arrays[key] = scatter(values)
    arrays.update(pulse_arrays)
    return arrays


def _to_flat(frame: Frame) -> dict:
    arrays = {k: np.ascontiguousarray(frame.xyz[:, col]) for col, k in enumerate("xyz")}
    echo_key = frame.meta.get("echo_key") or "echo_number"
    for key, values in frame.fields.items():
        if key == "echo":
            dtype = frame.meta.get("echo_dtype")
            arrays[echo_key] = values.astype(dtype) if dtype else values
        else:
            arrays[key] = values
    return arrays
