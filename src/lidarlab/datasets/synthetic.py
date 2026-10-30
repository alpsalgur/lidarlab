"""Synthetic multi-echo LiDAR recordings.

Simulates one spinning sensor on a car driving down a city street and writes one ``.npz``
file per frame. The echo convention matches the target data:

- echo 0: reflection from the sensor cover, a few cm in front of the optics
- echo 1: first real return; can be rain or spray in front of the actual target
- echo 2: second real return: the target behind clutter, tree crowns, object edges

Layouts:

- ``echo-axis``: per-point keys have shape (3, P), row = echo, NaN = no return;
  ``ring`` belongs to the pulse and has shape (P,)
- ``flat``: valid points only, shape (M,), plus ``echo_number``

Both carry the frame-level scalars ``roll``, ``pitch``, ``yaw`` (ego orientation, rad) and
``timestamp`` (POSIX s). Points are in the sensor frame: x forward, y left, z up.

The weather changes so that filters have something to find: the first 35 % of the frames
are dry, then rain builds up until 60 % and stays. Rain wets the cover (more echo 0),
adds drops close to the sensor and makes the truck ahead throw up spray.

    python -m lidarlab.datasets.synthetic data/synthetic/city_rain --frames 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

N_ECHOES = 3
LANE_Y = -1.75  # the ego car drives in the right lane of a road along +x
T0 = 1_793_361_600.0  # timestamp of frame 0: 2026-10-30 12:00 UTC


@dataclass
class Config:
    frames: int = 200
    rate_hz: float = 10.0
    speed: float = 12.0  # m/s
    n_beams: int = 128
    n_azimuth: int = 2048  # pulses per beam and revolution
    elev_min_deg: float = -25.0
    elev_max_deg: float = 15.0
    mount_height: float = 1.9
    max_range: float = 120.0
    range_noise: float = 0.02
    echo_separation: float = 0.8  # returns closer than this merge into one echo
    rain: float = 1.0  # peak rain level, 0 = dry all the way
    layout: str = "echo-axis"
    compress: bool = True
    seed: int = 0


@dataclass
class Scene:
    box_lo: np.ndarray  # (B, 3) axis-aligned boxes: buildings, parked cars, signs
    box_hi: np.ndarray
    box_refl: np.ndarray  # (B,) reflectivity, 0-1
    cyl: np.ndarray  # (C, 5) vertical cylinders: cx, cy, radius, z0, z1
    cyl_refl: np.ndarray
    crowns: np.ndarray  # (S, 4) tree crowns: cx, cy, cz, radius; partly see-through


def build_scene(cfg: Config, rng: np.random.Generator) -> Scene:
    """Street along +x lined with buildings, parked cars, street lights and trees."""
    x0 = -cfg.max_range - 20.0
    x1 = cfg.speed * cfg.frames / cfg.rate_hz + cfg.max_range + 20.0
    lo, hi, refl, cyl, cyl_refl, crowns = [], [], [], [], [], []

    def box(a, b, r):
        lo.append(a)
        hi.append(b)
        refl.append(r)

    for side in (-1.0, 1.0):
        x = x0
        while x < x1:  # buildings behind the sidewalk
            length, depth, height = rng.uniform(12, 40), rng.uniform(8, 15), rng.uniform(6, 22)
            y_a, y_b = sorted((side * 9.5, side * (9.5 + depth)))
            box([x, y_a, 0.0], [x + length, y_b, height], rng.uniform(0.2, 0.45))
            x += length + rng.uniform(2, 12)
        for x in np.arange(x0, x1, 6.5):  # parked cars
            if rng.random() < 0.55:
                length, height = rng.uniform(4.0, 4.8), rng.uniform(1.4, 1.7)
                y = side * 4.7
                box([x, y - 0.9, 0.2], [x + length, y + 0.9, height], rng.uniform(0.3, 0.6))
        for i, x in enumerate(np.arange(x0 + rng.uniform(0, 30), x1, 30.0)):  # street lights
            y = side * 6.2
            cyl.append([x, y, 0.15, 0.0, 7.0])
            cyl_refl.append(0.5)
            if i % 2 == 0:  # retro-reflective sign facing the traffic
                box([x - 0.05, y - 0.4, 2.4], [x + 0.05, y + 0.4, 3.2], 0.95)
        for x in np.arange(x0 + rng.uniform(0, 15), x1, 15.0):  # trees
            y, r = side * 7.6, rng.uniform(1.6, 2.6)
            cyl.append([x, y, 0.2, 0.0, 3.5])
            cyl_refl.append(0.3)
            crowns.append([x, y, 3.5 + 0.8 * r, r])

    def f32(a):
        return np.asarray(a, np.float32)

    return Scene(f32(lo), f32(hi), f32(refl), f32(cyl), f32(cyl_refl), f32(crowns))


def dynamic_objects(t: float, ego_x: float):
    """Truck ahead in the ego lane, oncoming cars, and the spray volume behind the truck."""
    truck_x = ego_x + 22.0 + 5.0 * np.sin(2 * np.pi * t / 20.0)
    lo = [[truck_x, LANE_Y - 1.25, 0.3]]
    hi = [[truck_x + 9.0, LANE_Y + 1.25, 3.8]]
    for start in (60.0, 160.0, 260.0, 360.0, 460.0):
        x = start - 14.0 * t
        lo.append([x, -LANE_Y - 0.95, 0.2])
        hi.append([x + 4.6, -LANE_Y + 0.95, 1.55])
    refl = [0.45] + [0.5] * 5
    spray = (
        np.array([truck_x - 10.0, LANE_Y - 1.6, 0.0], np.float32),
        np.array([truck_x, LANE_Y + 1.6, 2.8], np.float32),
    )
    return np.array(lo, np.float32), np.array(hi, np.float32), np.array(refl, np.float32), spray


def rain_level(cfg: Config, k: int) -> float:
    """Dry for the first 35 % of the frames, rain builds up until 60 %, then stays."""
    u = k / max(cfg.frames - 1, 1)
    return cfg.rain * float(np.clip((u - 0.35) / 0.25, 0.0, 1.0))


def ego_pose(t: float) -> tuple[float, float, float]:
    """Small roll, pitch and yaw wobble of the car body, rad."""
    roll = 0.004 * np.sin(2 * np.pi * t / 2.1)
    pitch = 0.006 * np.sin(2 * np.pi * t / 1.3) + 0.002 * np.sin(2 * np.pi * t / 0.37)
    yaw = 0.012 * np.sin(2 * np.pi * t / 7.0)
    return float(roll), float(pitch), float(yaw)


def rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def beam_directions(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors of all pulses in the sensor frame (beam-major order) and their ring."""
    el = np.deg2rad(np.linspace(cfg.elev_min_deg, cfg.elev_max_deg, cfg.n_beams))[:, None]
    az = 2 * np.pi * np.arange(cfg.n_azimuth)[None, :] / cfg.n_azimuth
    d = np.stack(
        np.broadcast_arrays(np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)),
        axis=-1,
    )
    ring = np.repeat(np.arange(cfg.n_beams, dtype=np.uint8), cfg.n_azimuth)
    return d.reshape(-1, 3).astype(np.float32), ring


def cover_probability(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Per-pulse chance of a cover echo: a low base level plus a few dirt spots."""
    beam = np.arange(cfg.n_beams)[:, None]
    az = np.arange(cfg.n_azimuth)[None, :]
    p = np.full((cfg.n_beams, cfg.n_azimuth), 0.08)
    for _ in range(6):
        c_beam, c_az = rng.uniform(0, cfg.n_beams), rng.uniform(0, cfg.n_azimuth)
        s_beam = rng.uniform(0.03, 0.11) * cfg.n_beams
        s_az = rng.uniform(0.015, 0.05) * cfg.n_azimuth
        d_az = np.abs(az - c_az)
        d_az = np.minimum(d_az, cfg.n_azimuth - d_az)  # azimuth wraps around
        p = p + 0.85 * np.exp(-0.5 * (((beam - c_beam) / s_beam) ** 2 + (d_az / s_az) ** 2))
    return np.clip(p, 0.0, 0.95).ravel()


def ground_reflectivity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Asphalt, sidewalk and lane markings (dashed centre line, solid edge lines)."""
    refl = np.where(np.abs(y) > 5.8, 0.25, 0.12).astype(np.float32)
    centre = (np.abs(y) < 0.08) & (np.mod(x, 12.0) < 6.0)
    edges = np.abs(np.abs(y) - 3.5) < 0.08
    refl[centre | edges] = 0.7
    return refl


def hit_ground(o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Distance to the ground plane z = 0 (inf for rays that do not point down)."""
    t = np.full(len(d), np.inf, np.float32)
    down = d[:, 2] < -1e-6
    t[down] = -o[2] / d[down, 2]
    return t


def hit_boxes(o, inv_d, lo, hi, chunk=32):
    """Nearest entry into axis-aligned boxes: distance, box index and entry axis."""
    n = len(inv_d)
    best_t = np.full(n, np.inf, np.float32)
    best_i = np.zeros(n, np.intp)
    if len(lo) == 0:
        return best_t, best_i, np.zeros(n, np.intp)
    rows = np.arange(n)
    for s in range(0, len(lo), chunk):
        # slab test one axis at a time: (n, b) arrays, no reductions over a size-3 axis
        t_near = t_far = None
        for ax in range(3):
            t1 = (lo[s : s + chunk, ax] - o[ax]) * inv_d[:, ax, None]
            t2 = (hi[s : s + chunk, ax] - o[ax]) * inv_d[:, ax, None]
            if t_near is None:
                t_near, t_far = np.minimum(t1, t2), np.maximum(t1, t2)
            else:
                t_near = np.maximum(t_near, np.minimum(t1, t2))
                t_far = np.minimum(t_far, np.maximum(t1, t2))
        t_near[(t_near > t_far) | (t_near <= 0)] = np.inf
        j = t_near.argmin(axis=1)
        t = t_near[rows, j]
        better = t < best_t
        best_t[better] = t[better]
        best_i[better] = s + j[better]
    # the axis of the entry slab of the chosen box gives the face normal
    entry = np.minimum((lo[best_i] - o) * inv_d, (hi[best_i] - o) * inv_d)
    return best_t, best_i, entry.argmax(axis=1)


def hit_cylinders(o, d, cyl):
    """Nearest hit on vertical cylinders (cx, cy, radius, z0, z1): distance and index."""
    n = len(d)
    if len(cyl) == 0:
        return np.full(n, np.inf, np.float32), np.zeros(n, np.intp)
    ox, oy = o[0] - cyl[:, 0], o[1] - cyl[:, 1]
    a = d[:, :1] ** 2 + d[:, 1:2] ** 2
    b = 2.0 * (d[:, :1] * ox + d[:, 1:2] * oy)
    c = ox**2 + oy**2 - cyl[:, 2] ** 2
    disc = b * b - 4.0 * a * c
    t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
    z = o[2] + t * d[:, 2:3]
    t[(disc < 0) | (t <= 0) | (z < cyl[:, 3]) | (z > cyl[:, 4])] = np.inf
    j = t.argmin(axis=1)
    return t[np.arange(n), j], j


def hit_spheres(o, d, sph):
    """Entry and exit distance through the nearest sphere (cx, cy, cz, radius)."""
    n = len(d)
    if len(sph) == 0:
        return np.full(n, np.inf, np.float32), np.full(n, np.inf, np.float32)
    oc = o - sph[:, :3]
    b = d @ oc.T
    c = (oc**2).sum(axis=1) - sph[:, 3] ** 2
    disc = b * b - c
    sq = np.sqrt(np.maximum(disc, 0.0))
    t_out = -b + sq
    t_in = np.maximum(-b - sq, 0.0)
    t_in[(disc <= 0) | (t_out <= 0)] = np.inf
    j = t_in.argmin(axis=1)
    rows = np.arange(n)
    return t_in[rows, j], t_out[rows, j]


def box_span(o, inv_d, lo, hi):
    """Entry and exit distance of every ray through one box (inf where it misses)."""
    t1 = (lo - o) * inv_d
    t2 = (hi - o) * inv_d
    t_in = np.maximum(np.minimum(t1, t2).max(axis=1), 0.0)
    t_out = np.maximum(t1, t2).min(axis=1)
    miss = t_in >= t_out
    t_in[miss] = np.inf
    t_out[miss] = np.inf
    return t_in, t_out


def assemble_echoes(cand_range, cand_int, separation):
    """Sort return candidates by range, merge ones closer than ``separation``, keep two."""
    order = np.argsort(cand_range, axis=1)
    cand_range = np.take_along_axis(cand_range, order, axis=1)
    cand_int = np.take_along_axis(cand_int, order, axis=1)
    last = cand_range[:, 0].copy()
    with np.errstate(invalid="ignore"):
        for j in range(1, cand_range.shape[1]):
            merged = cand_range[:, j] - last < separation
            cand_range[merged, j] = np.inf
            last = np.where(np.isfinite(cand_range[:, j]), cand_range[:, j], last)
    order = np.argsort(cand_range, axis=1)[:, :2]
    return np.take_along_axis(cand_range, order, axis=1), np.take_along_axis(cand_int, order, 1)


def simulate_frame(cfg: Config, scene: Scene, dirs: np.ndarray, ring: np.ndarray,
                   cover_p: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Ray-cast frame ``k`` and return its arrays in ``cfg.layout``."""
    rng = np.random.default_rng((cfg.seed, k + 1))  # per frame: same result in any order
    t = k / cfg.rate_hz
    rain = rain_level(cfg, k)
    roll, pitch, yaw = ego_pose(t)
    ego_x = cfg.speed * t
    o = np.array([ego_x, LANE_Y, cfg.mount_height], np.float32)
    d = dirs @ rotation(roll, pitch, yaw).T.astype(np.float32)  # ray directions, world frame
    inv_d = 1.0 / np.where(np.abs(d) < 1e-9, np.float32(1e-9), d)
    n = len(d)
    rows = np.arange(n)

    # objects within reach of the sensor
    dyn_lo, dyn_hi, dyn_refl, spray = dynamic_objects(t, ego_x)
    lo = np.vstack([scene.box_lo, dyn_lo])
    hi = np.vstack([scene.box_hi, dyn_hi])
    box_refl = np.concatenate([scene.box_refl, dyn_refl])
    reach = cfg.max_range + 10.0
    near = (hi[:, 0] > ego_x - reach) & (lo[:, 0] < ego_x + reach)
    lo, hi, box_refl = lo[near], hi[near], box_refl[near]
    near = np.abs(scene.cyl[:, 0] - ego_x) < reach
    cyl, cyl_refl = scene.cyl[near], scene.cyl_refl[near]
    crowns = scene.crowns[np.abs(scene.crowns[:, 0] - ego_x) < reach]

    # nearest solid surface: ground, box or cylinder
    t_box, i_box, axis_box = hit_boxes(o, inv_d, lo, hi)
    t_cyl, i_cyl = hit_cylinders(o, d, cyl)
    t_all = np.stack([hit_ground(o, d), t_box, t_cyl], axis=1)
    kind = t_all.argmin(axis=1)
    t_solid = t_all[rows, kind]
    t_solid[t_solid > cfg.max_range] = np.inf
    hit = np.isfinite(t_solid)

    refl = np.zeros(n, np.float32)
    cos = np.zeros(n, np.float32)  # cosine of the incidence angle
    idx = np.flatnonzero(hit & (kind == 0))
    p = o + d[idx] * t_solid[idx, None]
    refl[idx] = ground_reflectivity(p[:, 0], p[:, 1])
    cos[idx] = -d[idx, 2]
    idx = np.flatnonzero(hit & (kind == 1))
    refl[idx] = box_refl[i_box[idx]]
    cos[idx] = np.abs(d[idx, axis_box[idx]])
    idx = np.flatnonzero(hit & (kind == 2))
    c = cyl[i_cyl[idx]]
    normal = (o[:2] + d[idx, :2] * t_solid[idx, None] - c[:, :2]) / c[:, 2:3]
    refl[idx] = cyl_refl[i_cyl[idx]]
    cos[idx] = np.abs((d[idx, :2] * normal).sum(axis=1))

    # return candidates per pulse: solid, edge background, crown, rain, spray
    cand_range = np.full((n, 5), np.inf, np.float32)
    cand_int = np.zeros((n, 5), np.float32)
    cand_range[:, 0] = t_solid
    falloff = 1.0 - 0.4 * np.minimum(t_solid, cfg.max_range) / cfg.max_range
    cand_int[:, 0] = np.clip(
        255.0 * refl * (0.3 + 0.7 * cos) * falloff * rng.uniform(0.9, 1.1, n), 0, 255
    )

    # object edges: part of the beam footprint passes the edge and hits the background
    img_r = t_solid.reshape(cfg.n_beams, cfg.n_azimuth)
    img_i = cand_int[:, 0].reshape(cfg.n_beams, cfg.n_azimuth)
    bg_r = np.full_like(img_r, np.inf)
    bg_i = np.zeros_like(img_i)
    with np.errstate(invalid="ignore"):
        for shift in (1, -1):
            nb_r = np.roll(img_r, shift, axis=1)
            edge = np.isfinite(img_r) & np.isfinite(nb_r) & (nb_r - img_r > 1.0) & np.isinf(bg_r)
            bg_r[edge] = nb_r[edge]
            bg_i[edge] = np.roll(img_i, shift, axis=1)[edge]
    idx = np.flatnonzero(np.isfinite(bg_r.ravel()) & (rng.random(n) < 0.35))
    cand_range[idx, 1] = bg_r.ravel()[idx]
    cand_int[idx, 1] = 0.6 * bg_i.ravel()[idx]

    # tree crowns: some beams return from the leaves, some pass, some are blocked
    t_in, t_out = hit_spheres(o, d, crowns)
    idx = np.flatnonzero((t_in < t_solid) & (t_in < cfg.max_range) & (rng.random(n) < 0.55))
    cand_range[idx, 2] = t_in[idx] + rng.random(idx.size) * 0.35 * (t_out[idx] - t_in[idx])
    cand_int[idx, 2] = 255.0 * 0.35 * rng.uniform(0.3, 1.0, idx.size)
    cand_range[idx[rng.random(idx.size) < 0.3], :2] = np.inf

    if rain > 0:
        # rain drops, mostly close to the sensor
        idx = np.flatnonzero(rng.random(n) < 0.012 * rain)
        r = 0.8 + rng.exponential(5.0, idx.size)
        keep = r < np.minimum(t_solid[idx], 40.0)
        cand_range[idx[keep], 3] = r[keep]
        cand_int[idx[keep], 3] = rng.uniform(1.0, 12.0, keep.sum())
        # spray thrown up behind the truck
        s_in, s_out = box_span(o, inv_d, *spray)
        idx = np.flatnonzero(np.isfinite(s_in))
        length = s_out[idx] - s_in[idx]
        inside = rng.random(idx.size) < 1.0 - np.exp(-0.25 * rain * length)
        idx, length = idx[inside], length[inside]
        r = s_in[idx] + rng.random(idx.size) * length
        keep = r < t_solid[idx]
        cand_range[idx[keep], 4] = r[keep]
        cand_int[idx[keep], 4] = rng.uniform(2.0, 18.0, keep.sum())

    echo_r, echo_i = assemble_echoes(cand_range, cand_int, cfg.echo_separation)

    ranges = np.full((N_ECHOES, n), np.nan, np.float32)
    inten = np.full((N_ECHOES, n), np.nan, np.float32)
    cover = np.flatnonzero(rng.random(n) < np.minimum(cover_p + 0.25 * rain, 0.95))
    ranges[0, cover] = rng.uniform(0.04, 0.09, cover.size)
    inten[0, cover] = rng.uniform(40.0, 220.0, cover.size)
    for e in (1, 2):
        idx = np.flatnonzero(np.isfinite(echo_r[:, e - 1]))
        ranges[e, idx] = echo_r[idx, e - 1] + rng.normal(0.0, cfg.range_noise, idx.size)
        inten[e, idx] = echo_i[idx, e - 1]
    xyz = dirs * ranges[..., None]  # (3, n, 3) in the sensor frame, NaN where no return

    meta = {
        "roll": np.float32(roll),
        "pitch": np.float32(pitch),
        "yaw": np.float32(yaw),
        "timestamp": np.float64(T0 + t),
    }
    if cfg.layout == "echo-axis":
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        return {"x": x, "y": y, "z": z, "intensity": inten, "ring": ring, **meta}
    valid = np.isfinite(inten)
    echo = np.broadcast_to(np.arange(N_ECHOES, dtype=np.uint8)[:, None], valid.shape)
    return {
        "x": xyz[..., 0][valid],
        "y": xyz[..., 1][valid],
        "z": xyz[..., 2][valid],
        "intensity": inten[valid],
        "echo_number": echo[valid],
        "ring": np.broadcast_to(ring, valid.shape)[valid],
        **meta,
    }


_state: tuple | None = None  # per process: (cfg, scene, dirs, ring, cover_p, out)


def _init_worker(*state) -> None:
    global _state
    _state = state


def _write_frame(k: int) -> None:
    cfg, scene, dirs, ring, cover_p, out = _state
    save = np.savez_compressed if cfg.compress else np.savez
    save(out / f"frame_{k:06d}.npz", **simulate_frame(cfg, scene, dirs, ring, cover_p, k))


def _log(msg: str) -> None:
    print(msg, flush=True)


def generate(out_dir: str | Path, cfg: Config, workers: int = 1, log=_log) -> Path:
    """Write ``cfg.frames`` frames and a ``recording.json`` description into ``out_dir``."""
    if cfg.layout not in ("echo-axis", "flat"):
        raise ValueError(f"unknown layout {cfg.layout!r}")
    if not 1 <= cfg.n_beams <= 255:
        raise ValueError("n_beams must be 1-255 (ring is stored as uint8)")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirs, ring = beam_directions(cfg)
    scene_rng = np.random.default_rng(cfg.seed)
    state = (cfg, build_scene(cfg, scene_rng), dirs, ring, cover_probability(cfg, scene_rng), out)
    start = time.perf_counter()

    def progress(done: int) -> None:
        if log and (done % 20 == 0 or done == cfg.frames):
            log(f"{done}/{cfg.frames} frames, {time.perf_counter() - start:.0f} s")

    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=state) as pool:
            for done, _ in enumerate(pool.map(_write_frame, range(cfg.frames)), 1):
                progress(done)
    else:
        _init_worker(*state)
        for k in range(cfg.frames):
            _write_frame(k)
            progress(k + 1)

    info = {
        "dataset": "synthetic",
        "generator": "lidarlab.datasets.synthetic",
        "layout": cfg.layout,
        "echoes": {
            "0": "sensor cover reflection",
            "1": "first real return",
            "2": "second real return",
        },
        "units": {
            "xyz": "m, sensor frame (x forward, y left, z up)",
            "roll/pitch/yaw": "rad, ego orientation",
            "timestamp": "s (POSIX)",
            "intensity": "0-255",
        },
        "config": asdict(cfg),
    }
    (out / "recording.json").write_text(json.dumps(info, indent=2))
    return out


def main(argv: list[str] | None = None) -> None:
    default = Config()
    ap = argparse.ArgumentParser(description="Generate a synthetic multi-echo LiDAR recording.")
    ap.add_argument("out", type=Path, help="output folder")
    ap.add_argument("--frames", type=int, default=default.frames)
    ap.add_argument("--beams", type=int, default=default.n_beams)
    ap.add_argument("--azimuth", type=int, default=default.n_azimuth,
                    help="pulses per beam and revolution (default: %(default)s)")
    ap.add_argument("--rain", type=float, default=default.rain, help="peak rain level, 0 = dry")
    ap.add_argument("--layout", choices=("echo-axis", "flat"), default=default.layout)
    ap.add_argument("--no-compress", action="store_true", help="larger files, faster loading")
    ap.add_argument("--seed", type=int, default=default.seed)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args(argv)
    cfg = Config(frames=a.frames, n_beams=a.beams, n_azimuth=a.azimuth, rain=a.rain,
                 layout=a.layout, compress=not a.no_compress, seed=a.seed)
    generate(a.out, cfg, workers=a.workers)
    print(f"wrote {cfg.frames} frames to {a.out}")


if __name__ == "__main__":
    main()
