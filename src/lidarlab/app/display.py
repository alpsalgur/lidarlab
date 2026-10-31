"""Display settings and the per-frame values the view colors points by (no Qt)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lidarlab import Frame

SOLID = "solid color"
COMPUTED = ("range", "height")
CATEGORICAL = ("echo", "label")  # one color per value, named in the legend
COLORMAPS = ("turbo", "viridis", "plasma", "inferno", "magma", "cividis", "jet", "rainbow", "gray",
             "coolwarm", "Spectral", "twilight")
DEFAULT_CMAPS = {"intensity": "turbo", "reflectivity": "turbo", "range": "viridis",
                 "height": "plasma", "ring": "jet"}


@dataclass
class Display:
    """How the current frame is drawn; the Properties panel and ``view`` edit it."""

    color_by: str = "intensity"
    cmaps: dict[str, str] = field(default_factory=dict)
    clims: dict[str, tuple[float, float]] = field(default_factory=dict)
    point_size: float = 2.0
    hidden_echoes: set[int] = field(default_factory=set)
    show_removed: bool = False
    split: bool = False  # raw points left, pipeline output right
    legend: bool = True
    rings: bool = True
    grid: bool = False
    ego: bool = True
    axes: bool = True
    background: str = "Dark gradient"

    def cmap(self, mode: str) -> str:
        return self.cmaps.get(mode, DEFAULT_CMAPS.get(mode, "viridis"))


def color_modes(frame: Frame) -> list[str]:
    """Color-by choices: the numeric per-point fields, range, height and a solid color."""
    names = [k for k, v in frame.fields.items() if v.ndim == 1 and v.dtype.kind in "biuf"]
    return [*names, *(m for m in COMPUTED if m not in names), SOLID]


def color_values(frame: Frame, mode: str) -> np.ndarray | None:
    """Per-point values for ``mode``; None for a solid color or a field the frame lacks."""
    if mode == "range" and "range" not in frame.fields:
        return frame.range.astype(np.float32, copy=False)
    if mode == "height":
        return np.ascontiguousarray(frame.z)
    values = frame.fields.get(mode)
    if values is None or values.ndim != 1 or values.dtype.kind not in "biuf":
        return None
    if mode in CATEGORICAL:
        return values.astype(np.int32)
    return values if values.dtype.kind == "f" else values.astype(np.float32)


def auto_range(values: np.ndarray) -> tuple[float, float]:
    """Color range: full range for whole numbers, else the 1st-99th percentile."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    if np.array_equal(finite, np.round(finite)):
        lo, hi = float(finite.min()), float(finite.max())
    else:
        lo, hi = (float(v) for v in np.percentile(finite, [1, 99]))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def removed_points(raw: Frame, out: Frame) -> Frame:
    """Points of ``raw`` whose index no longer appears in ``out``."""
    if not len(raw):
        return raw
    space = int(raw.index.max()) + 1
    kept = np.zeros(space, bool)
    idx = out.index
    kept[idx[(idx >= 0) & (idx < space)]] = True
    return raw[~kept[raw.index]]


def estimate_ground(frame: Frame, default: float = -1.8) -> float:
    """Ground height below the sensor: a low percentile of z between 3 and 30 m."""
    r = np.hypot(frame.x, frame.y)
    z = frame.z[(r > 3.0) & (r < 30.0)]
    return float(np.percentile(z, 10)) if z.size > 100 else default
