"""One LiDAR scan as a set of per-point arrays."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

_COORDS = {"x": 0, "y": 1, "z": 2}
_DERIVED = ("range", "azimuth", "elevation", "pulse")


class Frame:
    """One scan: ``xyz`` (N, 3) float32 plus named per-point fields of length N.

    Fields read like attributes or keys: ``frame.intensity``, ``frame["echo"]``. Derived
    values are computed on access: ``range`` (m), ``azimuth`` and ``elevation`` (deg) and,
    for echo-axis files, ``pulse`` (points with the same pulse are echoes of one shot).

    Indexing with a boolean mask, an index array or a slice returns a new Frame with every
    field subset; an integer returns one point as a dict. ``index`` keeps each point's
    position in the source file, so the points a processing step removed can always be
    found again. ``meta`` holds per-frame values such as ``timestamp``, ``roll``, ``pitch``,
    ``yaw`` and the frame number (``meta["frame"]``).

    Frames from a recording share read-only arrays with its cache. Assigning a field
    (``frame["intensity"] = frame.intensity * 2``) replaces the array in this Frame only;
    ``frame.copy()`` gives writable arrays.
    """

    __slots__ = ("xyz", "fields", "meta", "index")

    def __init__(self, xyz, fields: Mapping[str, np.ndarray] | None = None,
                 meta: Mapping | None = None, index=None):
        xyz = np.asarray(xyz)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")
        if xyz.dtype != np.float32 or not xyz.flags.c_contiguous:
            xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        n = len(xyz)
        checked = {}
        for name, values in (fields or {}).items():
            values = np.asarray(values)
            if values.shape[:1] != (n,):
                raise ValueError(f"field {name!r} has shape {values.shape} for {n} points")
            checked[name] = values
        index = np.arange(n, dtype=np.int32) if index is None else np.asarray(index)
        if index.shape != (n,):
            raise ValueError(f"index has shape {index.shape} for {n} points")
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "fields", checked)
        object.__setattr__(self, "meta", dict(meta or {}))
        object.__setattr__(self, "index", index)

    def __reduce__(self):
        return Frame, (self.xyz, self.fields, self.meta, self.index)

    # fields as attributes

    def __getattr__(self, name):
        try:
            fields = object.__getattribute__(self, "fields")
        except AttributeError:  # not initialised yet
            raise AttributeError(name) from None
        if name in fields:
            return fields[name]
        raise AttributeError(f"Frame has no field {name!r} (fields: {', '.join(fields) or 'none'})")

    def __setattr__(self, name, value):
        if name in ("fields", "meta", "index"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __dir__(self):
        return sorted({*super().__dir__(), *self.fields})

    # container protocol

    def __len__(self) -> int:
        return len(self.xyz)

    def __contains__(self, name) -> bool:
        return name in self.fields or name == "xyz" or name in _COORDS

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "xyz":
                return self.xyz
            if key in _COORDS:
                return self.xyz[:, _COORDS[key]]
            if key in self.fields:
                return self.fields[key]
            if key in _DERIVED:
                return getattr(self, key)
            raise KeyError(f"no field {key!r} (fields: {', '.join(self.fields) or 'none'})")
        if isinstance(key, int | np.integer):
            i = int(key)
            point = dict(zip("xyz", self.xyz[i].tolist(), strict=True))
            point.update((name, values[i].tolist()) for name, values in self.fields.items())
            return point
        if isinstance(key, list):
            key = np.asarray(key)
        if isinstance(key, np.ndarray) and key.dtype == bool and key.shape != (len(self),):
            raise IndexError(f"mask has shape {key.shape} for {len(self)} points")
        return Frame(self.xyz[key], {k: v[key] for k, v in self.fields.items()}, self.meta,
                     self.index[key])

    def __setitem__(self, key, values) -> None:
        if not isinstance(key, str):
            raise TypeError("assign fields by name: frame['name'] = values")
        n = len(self)
        if key == "xyz":
            values = np.ascontiguousarray(values, dtype=np.float32)
            if values.shape != (n, 3):
                raise ValueError(f"xyz must have shape ({n}, 3), got {values.shape}")
            object.__setattr__(self, "xyz", values)
            return
        if key in _COORDS:
            xyz = self.xyz.copy()
            xyz[:, _COORDS[key]] = values
            object.__setattr__(self, "xyz", xyz)
            return
        values = np.asarray(values)
        if values.ndim == 0:
            values = np.full(n, values)
        if values.shape[:1] != (n,):
            raise ValueError(f"field {key!r} needs {n} values, got shape {values.shape}")
        self.fields[key] = values

    def __delitem__(self, key: str) -> None:
        del self.fields[key]

    def __repr__(self) -> str:
        parts = [f"{len(self):,} points"]
        if self.fields:
            parts.append("fields: " + ", ".join(self.fields))
        frame_no = self.meta.get("frame")
        head = "Frame" if frame_no is None else f"Frame #{frame_no}"
        return f"{head}({'; '.join(parts)})"

    # coordinates and derived values

    @property
    def x(self) -> np.ndarray:
        return self.xyz[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xyz[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.xyz[:, 2]

    @property
    def range(self) -> np.ndarray:
        """Distance from the sensor, m."""
        if "range" in self.fields:
            return self.fields["range"]
        return np.sqrt(np.einsum("ij,ij->i", self.xyz, self.xyz))

    @property
    def azimuth(self) -> np.ndarray:
        """Horizontal angle, deg: 0 = forward (+x), positive to the left (+y)."""
        return np.degrees(np.arctan2(self.xyz[:, 1], self.xyz[:, 0]))

    @property
    def elevation(self) -> np.ndarray:
        """Vertical angle above the sensor's horizontal plane, deg."""
        return np.degrees(np.arctan2(self.xyz[:, 2], np.hypot(self.xyz[:, 0], self.xyz[:, 1])))

    @property
    def pulse(self) -> np.ndarray:
        """Pulse of each point; the echoes of one pulse share it (echo-axis files only)."""
        if "pulse" in self.fields:
            return self.fields["pulse"]
        shape = self.meta.get("shape")
        if shape is None:
            raise ValueError("pulse ids need echo-axis data (meta['shape'] = (echoes, pulses))")
        return self.index % shape[1]

    # copies and output

    def view(self) -> Frame:
        """New Frame sharing the arrays; adding or replacing fields leaves this one alone."""
        return Frame(self.xyz, self.fields, self.meta, self.index)

    def copy(self) -> Frame:
        """Deep copy with writable arrays."""
        return Frame(self.xyz.copy(), {k: v.copy() for k, v in self.fields.items()}, self.meta,
                     self.index.copy())

    def bounds(self) -> np.ndarray:
        """(2, 3) array: minimum and maximum of x, y, z (NaN for an empty frame)."""
        if not len(self):
            return np.full((2, 3), np.nan, np.float32)
        return np.stack([self.xyz.min(axis=0), self.xyz.max(axis=0)])

    def save(self, path, layout: str | None = None, compress: bool = True):
        """Write as .npz in the layout it was read from (or ``layout``); see lidarlab.core.npz."""
        from .npz import write_frame

        return write_frame(self, path, layout=layout, compress=compress)
