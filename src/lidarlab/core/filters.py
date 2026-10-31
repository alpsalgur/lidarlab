"""Built-in filters: echo selection, thresholds, cropping, leveling, downsampling, outlier
and ground removal.

Filters that remove points return a keep-mask, so every point keeps its source index and the
app can show what was removed. Use them in the app (Pipeline Browser > Add filter) or in
scripts:

    pipe.add(lab.filters.DROR(beta=3, alpha_deg=0.18))
    pipe["dror"].set(min_neighbors=2)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .filter import REGISTRY, Filter, Param
from .frame import Frame

KEEP_INSIDE, REMOVE_INSIDE = "keep inside", "remove inside"
INSIDE = (KEEP_INSIDE, REMOVE_INSIDE)

__all__ = ["DROR", "DSOR", "LIOR", "ROR", "SOR", "CropBox", "EchoSelect", "GroundPlane",
           "Level", "Threshold", "VoxelGrid", "WeakFirstEcho", "available"]


def available() -> dict[str, type[Filter]]:
    """Built-in filters by key."""
    return dict(REGISTRY)


def _echo(f: Frame) -> np.ndarray:
    """Echo numbers; data without echoes counts as echo 1 (a single real return)."""
    return f.fields["echo"] if "echo" in f.fields else np.ones(len(f), np.uint8)


def _pulse(f: Frame) -> np.ndarray | None:
    try:
        return f.pulse
    except ValueError:
        return None


def _first_in_group(groups: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Mask over ``groups``: the first element of each group when visited in ``order``."""
    g = groups[order]
    first = np.ones(len(g), bool)
    first[1:] = g[1:] != g[:-1]
    keep = np.zeros(len(groups), bool)
    keep[order[first]] = True
    return keep


def _has_neighbors(points: np.ndarray, radius, k: int, query: np.ndarray | None = None):
    """Mask: which query points (default: all) have at least ``k`` other points of ``points``
    within ``radius`` (scalar or one per query point).

    Looks at the distance of the k-th nearest neighbour instead of counting every point in the
    ball: same answer, and fast where points are dense. Query points must be in ``points``.
    """
    query = points if query is None else query
    if len(points) <= k or not len(query):
        return np.zeros(len(query), bool)
    # unbalanced trees build ~1.6x faster on scans; the bound cuts the search short
    bound = float(np.max(radius)) * (1 + 1e-6) + 1e-9
    d, _ = _tree(points).query(query, k + 1, distance_upper_bound=bound, workers=-1)
    return d[:, k] <= radius  # the nearest one is the point itself; inf beyond the bound


def _tree(points: np.ndarray) -> cKDTree:
    return cKDTree(points, balanced_tree=False)


def _mean_knn_distance(points: np.ndarray, k: int) -> np.ndarray:
    d, _ = _tree(points).query(points, k + 1, workers=-1)  # the nearest one is the point itself
    return d[:, 1:].mean(axis=1)


# echoes


class EchoSelect(Filter):
    """Keep echoes by their role. A pulse returns up to three echoes: 0 is the reflection from
    the sensor cover, 1 and 2 are real returns. "first", "last" and "strongest" keep one real
    echo per pulse and need pulse ids (echo-axis files); "only echo n" keeps one echo number.
    """

    key, title, group = "echo", "Echo selection", "Echo"
    MODES = ("drop cover echo", "first real echo", "last real echo", "strongest real echo",
             "only echo n")
    params = {
        "mode": Param("drop cover echo", choices=MODES),
        "n": Param(1, 0, 5, kind="int", help="echo kept by 'only echo n'"),
        "cover": Param(0, 0, 5, kind="int", help="echo number of cover reflections"),
    }

    def apply(self, f: Frame):
        echo = _echo(f)
        if self.mode == "only echo n":
            return echo == self.n
        real = echo != self.cover
        if self.mode == "drop cover echo":
            return real
        pulse = _pulse(f)
        if pulse is None:
            if len(np.unique(echo[real])) <= 1:
                return real  # one real echo per pulse anyway
            raise ValueError(f"'{self.mode}' needs pulse ids (echo-axis files)")
        idx = np.flatnonzero(real)
        if self.mode == "first real echo":
            rank = echo[idx].astype(np.int32)
        elif self.mode == "last real echo":
            rank = -echo[idx].astype(np.int32)
        else:
            rank = -f.intensity[idx]
        keep = np.zeros(len(f), bool)
        keep[idx] = _first_in_group(pulse[idx], np.lexsort((rank, pulse[idx])))
        return keep


class WeakFirstEcho(Filter):
    """Remove weak first echoes of pulses that also have a later echo: rain drops and spray
    in front of a target. The target returns the later echo and stays. ``max_intensity`` is
    in the units of the data (the synthetic run uses 0-255, KITTI 0-1).
    """

    key, title, group = "weak_first_echo", "Weak first echoes (rain, spray)", "Echo"
    params = {
        "max_intensity": Param(20.0, 0.0, help="first echoes weaker than this are removed"),
        "cover": Param(0, 0, 5, kind="int", help="echo number of cover reflections"),
    }

    def apply(self, f: Frame):
        echo = _echo(f)
        real = np.flatnonzero(echo != self.cover)
        pulse = _pulse(f)
        if pulse is None:
            if len(np.unique(echo[real])) <= 1:
                return None  # single echo data: nothing to compare
            raise ValueError("needs pulse ids (echo-axis files)")
        p = pulse[real]
        _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
        first = _first_in_group(p, np.lexsort((echo[real], p)))
        weak = first & (counts[inverse] > 1) & (f.intensity[real] < self.max_intensity)
        keep = np.ones(len(f), bool)
        keep[real[weak]] = False
        return keep


# basic selections


class Threshold(Filter):
    """Keep the points whose field lies within [min, max], or remove them. Besides the
    data fields, x, y, z, range, azimuth and elevation can be used."""

    key, title, group = "threshold", "Threshold", "Basic"
    params = {
        "field": Param("range", kind="field"),
        "min": Param(0.5),
        "max": Param(120.0),
        "mode": Param(KEEP_INSIDE, choices=INSIDE),
    }

    def apply(self, f: Frame):
        values = f[self.field]
        inside = (values >= self.min) & (values <= self.max)
        return inside if self.mode == KEEP_INSIDE else ~inside


class CropBox(Filter):
    """Keep the points inside an axis-aligned box in the sensor frame, or remove them
    (e.g. reflections from the own vehicle)."""

    key, title, group = "crop_box", "Crop box", "Basic"
    params = {
        "x_min": Param(-60.0, unit="m"), "x_max": Param(100.0, unit="m"),
        "y_min": Param(-40.0, unit="m"), "y_max": Param(40.0, unit="m"),
        "z_min": Param(-4.0, unit="m"), "z_max": Param(6.0, unit="m"),
        "mode": Param(KEEP_INSIDE, choices=INSIDE),
    }

    def apply(self, f: Frame):
        lo = np.array([self.x_min, self.y_min, self.z_min], np.float32)
        hi = np.array([self.x_max, self.y_max, self.z_max], np.float32)
        inside = np.all((f.xyz >= lo) & (f.xyz <= hi), axis=1)
        return inside if self.mode == KEEP_INSIDE else ~inside


# geometry


def rotation(roll: float, pitch: float) -> np.ndarray:
    """Ry(pitch) @ Rx(roll): turns sensor coordinates into level ones (yaw unchanged)."""
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return ry @ rx


class Level(Filter):
    """Rotate the points by the frame's roll and pitch (``frame.meta``) so the ground is
    level; the heading stays. Removes nothing."""

    key, title, group = "level", "Level with roll / pitch", "Geometry"
    params = {"roll": Param(True), "pitch": Param(True)}

    def apply(self, f: Frame):
        missing = [k for k in ("roll", "pitch") if k not in f.meta]
        if missing:
            raise ValueError(f"needs {' and '.join(missing)} in frame.meta")
        r = rotation(float(f.meta["roll"]) if self.roll else 0.0,
                     float(f.meta["pitch"]) if self.pitch else 0.0)
        out = f.view()
        out["xyz"] = f.xyz @ r.T.astype(np.float32)
        return out


class VoxelGrid(Filter):
    """Downsample: keep one point (the first) per cubic voxel."""

    key, title, group = "voxel", "Voxel downsample", "Geometry"
    params = {"size": Param(0.2, 0.02, 2.0, 0.01, unit="m")}

    def apply(self, f: Frame):
        keep = np.zeros(len(f), bool)
        if not len(f):
            return keep
        cell = np.floor(f.xyz / self.size).astype(np.int64)
        cell -= cell.min(axis=0)
        dims = cell.max(axis=0) + 1
        linear = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
        _, first = np.unique(linear, return_index=True)
        keep[first] = True
        return keep


# outlier removal


class SOR(Filter):
    """Statistical outlier removal: drop points whose mean distance to their k nearest
    neighbours is above the frame's mean + std_ratio × std. Sparse far points suffer too,
    which DSOR corrects."""

    key, title, group = "sor", "SOR – statistical outliers", "Noise"
    params = {
        "k": Param(8, 1, 50, kind="int", help="neighbours"),
        "std_ratio": Param(2.0, 0.0, 10.0, 0.1),
    }

    def apply(self, f: Frame):
        if len(f) <= self.k:
            return None
        mean = _mean_knn_distance(f.xyz, self.k)
        return mean <= mean.mean() + self.std_ratio * mean.std()


class ROR(Filter):
    """Radius outlier removal: drop points with fewer than min_neighbors other points within
    a fixed radius. Only suits near points, as the spacing grows with range (see DROR)."""

    key, title, group = "ror", "ROR – radius outliers", "Noise"
    params = {
        "radius": Param(0.4, 0.02, 3.0, 0.01, unit="m"),
        "min_neighbors": Param(2, 1, 30, kind="int"),
        "max_range": Param(30.0, 1.0, 250.0, 1.0, unit="m", help="farther points are kept"),
    }

    def apply(self, f: Frame):
        idx = np.flatnonzero(f.range < self.max_range)
        keep = np.ones(len(f), bool)
        keep[idx] = _has_neighbors(f.xyz[idx], self.radius, self.min_neighbors)
        return keep


class DROR(Filter):
    """Dynamic radius outlier removal (Charron et al. 2018): like ROR, but the search radius
    grows with the horizontal distance r, radius = beta · r · alpha, at least min_radius.
    alpha is the sensor's horizontal angular resolution. Made for snow; works on rain drops
    and spray, which have few neighbours at any range. min_neighbors defaults to 2 (the paper
    uses 3): on the synthetic run it catches 95 % of the rain with a third of the false hits."""

    key, title, group = "dror", "DROR – dynamic radius outliers", "Noise"
    params = {
        "beta": Param(3.0, 0.5, 10.0, 0.1, help="radius multiplier"),
        "alpha_deg": Param(0.2, 0.01, 2.0, 0.01, unit="°", help="horizontal resolution"),
        "min_radius": Param(0.04, 0.0, 1.0, 0.01, unit="m"),
        "min_neighbors": Param(2, 1, 30, kind="int"),
        "max_range": Param(80.0, 1.0, 250.0, 1.0, unit="m", help="farther points are kept"),
    }

    def apply(self, f: Frame):
        idx = np.flatnonzero(f.range < self.max_range)
        keep = np.ones(len(f), bool)
        points = f.xyz[idx]
        r_xy = np.hypot(points[:, 0], points[:, 1])
        radius = np.maximum(self.beta * r_xy * np.radians(self.alpha_deg), self.min_radius)
        keep[idx] = _has_neighbors(points, radius, self.min_neighbors)
        return keep


class DSOR(Filter):
    """Dynamic statistical outlier removal (Kurup & Bos 2021): SOR whose threshold grows with
    range, threshold = (mean + std_mult · std) · range_mult · range. The paper's
    range_mult 0.05 suits a 32-beam sensor; denser clouds need more (default 0.1)."""

    key, title, group = "dsor", "DSOR – dynamic statistical outliers", "Noise"
    params = {
        "k": Param(5, 1, 50, kind="int", help="neighbours"),
        "std_mult": Param(0.01, 0.0, 5.0, 0.01),
        "range_mult": Param(0.1, 0.005, 1.0, 0.005),
    }

    def apply(self, f: Frame):
        if len(f) <= self.k:
            return None
        mean = _mean_knn_distance(f.xyz, self.k)
        global_threshold = mean.mean() + self.std_mult * mean.std()
        return mean <= global_threshold * self.range_mult * f.range


class LIOR(Filter):
    """Low-intensity outlier removal (Park et al. 2020): snow and rain reflect weakly, so only
    points below max_intensity are checked with a radius outlier test. ``max_intensity`` is
    in the units of the data."""

    key, title, group = "lior", "LIOR – low-intensity outliers", "Noise"
    params = {
        "max_intensity": Param(20.0, 0.0),
        "radius": Param(0.3, 0.02, 3.0, 0.01, unit="m"),
        "min_neighbors": Param(3, 1, 30, kind="int"),
        "max_range": Param(40.0, 1.0, 250.0, 1.0, unit="m", help="farther points are kept"),
    }

    def apply(self, f: Frame):
        near = f.range < self.max_range
        low = np.flatnonzero(near & (f.intensity < self.max_intensity))
        keep = np.ones(len(f), bool)
        if len(low):
            keep[low] = _has_neighbors(f.xyz[near], self.radius, self.min_neighbors, f.xyz[low])
        return keep


# ground


class GroundPlane(Filter):
    """Find the ground as a plane with RANSAC on the low points within max_range, then
    remove the points within ``distance`` of it (or keep only those). Planes steeper than
    max_slope_deg are rejected, so walls do not win."""

    key, title, group = "ground", "Ground plane (RANSAC)", "Ground"
    params = {
        "distance": Param(0.2, 0.02, 1.0, 0.01, unit="m"),
        "max_range": Param(40.0, 5.0, 150.0, 1.0, unit="m"),
        "max_slope_deg": Param(10.0, 0.0, 45.0, 1.0, unit="°"),
        "iterations": Param(100, 10, 1000, kind="int"),
        "mode": Param("remove ground", choices=("remove ground", "keep ground")),
    }

    def plane(self, f: Frame) -> tuple[np.ndarray, float]:
        """Unit normal n (pointing up) and offset d of the plane n·p + d = 0."""
        rng = np.random.default_rng(int(f.meta.get("frame", 0)))  # same result per frame
        r = f.range
        cand = np.flatnonzero((r > 2.0) & (r < self.max_range))
        if len(cand) < 3:
            raise ValueError(f"fewer than 3 points within {self.max_range} m")
        z = f.z[cand]
        cand = cand[z <= np.median(z)]  # the ground is below most of the scene
        sample = f.xyz[rng.choice(cand, min(len(cand), 20000), replace=False)].astype(np.float64)
        tri = rng.integers(0, len(sample), (self.iterations, 3))
        a, b, c = sample[tri[:, 0]], sample[tri[:, 1]], sample[tri[:, 2]]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal, axis=1)
        valid = length > 1e-9
        normal[valid] /= length[valid, None]
        normal *= np.where(normal[:, 2:] < 0, -1.0, 1.0)
        valid &= normal[:, 2] >= np.cos(np.radians(self.max_slope_deg))
        if not valid.any():
            raise ValueError("no plane flat enough; raise max_slope_deg")
        offset = -(normal * a).sum(axis=1)
        inliers = np.abs(sample @ normal.T + offset) < self.distance
        support = np.where(valid, inliers.sum(axis=0), -1)
        best = inliers[:, int(np.argmax(support))]
        # least squares refinement on the inliers: z = u x + v y + w
        p = sample[best]
        (u, v, w), *_ = np.linalg.lstsq(np.c_[p[:, :2], np.ones(len(p))], p[:, 2], rcond=None)
        n = np.array([-u, -v, 1.0])
        scale = np.linalg.norm(n)
        return n / scale, -w / scale

    def apply(self, f: Frame):
        if not len(f):
            return None
        n, d = self.plane(f)
        ground = np.abs(f.xyz @ n.astype(np.float32) + np.float32(d)) < self.distance
        return ~ground if self.mode == "remove ground" else ground
