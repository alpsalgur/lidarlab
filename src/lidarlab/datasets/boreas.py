"""Boreas (Toronto, all seasons, snow) -> per-frame .npz files in the flat layout.

Downloads a stretch of one sequence from the public Boreas S3 bucket (no AWS account
needed): the lidar poses, the lidar-to-vehicle calibration and the chosen scans of the
128-beam Velodyne Alpha Prime (10 Hz, one return per pulse, ~215k points). Each scan becomes
one ``.npz``:

- ``x, y, z``: float32 m, sensor at the origin, axes turned to the vehicle (x forward,
  y left, z up). Boreas mounts the lidar turned by ~47° about z; the calibration undoes it.
- ``intensity``: float32 0-255; ``ring``: uint8 laser 0-127; ``t``: float32 s from the
  middle of the scan; ``echo_number``: uint8, always 1 (single return)
- ``roll, pitch, yaw``: float32 rad from the post-processed Applanix poses. Roll and pitch
  follow lidarlab's Level filter (``Ry(pitch) @ Rx(roll)`` levels the points), yaw is the
  heading of x (counter-clockwise from east).
- ``timestamp``: float64 POSIX s (middle of the scan)

Boreas is licensed CC BY 4.0. Cite: Burnett et al., "Boreas: A Multi-Season Autonomous
Driving Dataset", IJRR 2023.

    python -m lidarlab.datasets.boreas boreas-2021-01-26-11-22 data/boreas --start 3300
"""

from __future__ import annotations

import argparse
import http.client
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE_URL = "https://boreas.s3.amazonaws.com"
# Applanix frame (x right, y forward, z up) -> vehicle frame (x forward, y left, z up)
VEHICLE_FROM_APPLANIX = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _log(msg: str) -> None:
    print(msg, flush=True)


def fetch(url: str, path: Path, attempts: int = 4) -> Path:
    """Download ``url`` to ``path`` unless it is there already; network errors are retried."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            part.replace(path)
            return path
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == attempts:
                raise  # missing file or refused: retrying will not help
        except (OSError, http.client.HTTPException):
            if attempt == attempts:
                raise
        time.sleep(3 * attempt)
    return path


def read_poses(path: Path) -> np.ndarray:
    """lidar_poses.csv rows: t (us), x, y, z, vx, vy, vz, roll, pitch, heading, wz, wy, wx."""
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def _rx(r):
    return np.array([[1, 0, 0], [0, np.cos(r), np.sin(r)], [0, -np.sin(r), np.cos(r)]])


def _ry(p):
    return np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])


def _rz(y):
    return np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]])


def enu_from_lidar(roll: float, pitch: float, heading: float) -> np.ndarray:
    """Rotation of a lidar pose (pyboreas' yawPitchRollToRot): p_enu = C @ p_lidar."""
    return _rx(roll) @ _ry(pitch) @ _rz(heading)


def attitude(c_enu_vehicle: np.ndarray) -> tuple[float, float, float]:
    """(roll, pitch, yaw) of the vehicle frame for lidarlab: Ry(pitch) Rx(roll) turns the
    vehicle frame's "up" into +z; yaw is the heading of x from east."""
    up = c_enu_vehicle.T @ [0.0, 0.0, 1.0]
    roll = np.arctan2(up[1], up[2])
    pitch = np.arctan2(-up[0], np.hypot(up[1], up[2]))
    forward = c_enu_vehicle @ [1.0, 0.0, 0.0]
    return float(roll), float(pitch), float(np.arctan2(forward[1], forward[0]))


def download(sequence: str, raw_root: Path, start: int, frames: int, workers: int = 4,
             log=_log) -> Path:
    """Fetch poses, calibration and scans ``start`` .. ``start + frames - 1``."""
    src = raw_root / sequence
    base = f"{BASE_URL}/{sequence}"
    fetch(f"{base}/applanix/lidar_poses.csv", src / "applanix" / "lidar_poses.csv")
    fetch(f"{base}/calib/T_applanix_lidar.txt", src / "calib" / "T_applanix_lidar.txt")
    stamps = read_poses(src / "applanix" / "lidar_poses.csv")[:, 0].astype(np.int64)
    if not 0 <= start < len(stamps):
        raise ValueError(f"start must be 0-{len(stamps) - 1} ({len(stamps)} scans in {sequence})")
    wanted = stamps[start : start + frames]
    begin = time.perf_counter()

    def get(t):
        return fetch(f"{base}/lidar/{t}.bin", src / "lidar" / f"{t}.bin")

    with ThreadPoolExecutor(workers) as pool:
        for done, path in enumerate(pool.map(get, wanted), 1):
            if log and (done % 25 == 0 or done == len(wanted)):
                mb = path.stat().st_size * done / 1e6
                seconds = time.perf_counter() - begin
                log(f"{done}/{len(wanted)} scans (~{mb:,.0f} MB, {seconds:.0f} s)")
    return src


def convert(src: Path, out_dir: Path, start: int = 0, frames: int | None = None,
            compress: bool = True, log=_log) -> int:
    """Convert the downloaded scans of ``src`` (from pose row ``start`` on) to .npz frames."""
    poses = read_poses(src / "applanix" / "lidar_poses.csv")
    t_applanix_lidar = np.loadtxt(src / "calib" / "T_applanix_lidar.txt")
    vehicle_from_lidar = VEHICLE_FROM_APPLANIX @ t_applanix_lidar[:3, :3]
    rows = poses[start : None if frames is None else start + frames]
    rows = [r for r in rows if (src / "lidar" / f"{int(r[0])}.bin").exists()]
    if not rows:
        raise FileNotFoundError(f"no downloaded scans under {src / 'lidar'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compress else np.savez
    rot = vehicle_from_lidar.T.astype(np.float32)
    for k, row in enumerate(rows):
        stamp = int(row[0])
        pts = np.fromfile(src / "lidar" / f"{stamp}.bin", dtype=np.float32).reshape(-1, 6)
        xyz = pts[:, :3] @ rot
        c_enu_vehicle = enu_from_lidar(row[7], row[8], row[9]) @ vehicle_from_lidar.T
        roll, pitch, yaw = attitude(c_enu_vehicle)
        save(
            out_dir / f"frame_{k:06d}.npz",
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            intensity=pts[:, 3],
            ring=pts[:, 4].astype(np.uint8),
            t=pts[:, 5],
            echo_number=np.ones(len(pts), np.uint8),
            roll=np.float32(roll),
            pitch=np.float32(pitch),
            yaw=np.float32(yaw),
            timestamp=np.float64(stamp / 1e6),
        )
        if log and ((k + 1) % 50 == 0 or k + 1 == len(rows)):
            log(f"{k + 1}/{len(rows)} frames")
    info = {
        "dataset": "Boreas",
        "source": f"{src.name}, scans {start}-{start + len(rows) - 1}",
        "license": "CC BY 4.0 (Burnett et al., Boreas: A Multi-Season Autonomous Driving "
                   "Dataset, IJRR 2023)",
        "sensor": "Velodyne Alpha Prime, 128 beams, 10 Hz, single return",
        "layout": "flat",
        "echoes": {"1": "single return"},
        "units": {
            "xyz": "m, sensor at the origin, vehicle axes (x forward, y left, z up)",
            "roll/pitch/yaw": "rad, Applanix post-processed; yaw = heading from east",
            "timestamp": "s (POSIX), middle of the scan",
            "t": "s, point time relative to timestamp",
            "intensity": "0-255",
        },
        "frames": len(rows),
    }
    (out_dir / "recording.json").write_text(json.dumps(info, indent=2))
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download part of a Boreas sequence as .npz frames.")
    ap.add_argument("sequence", help="e.g. boreas-2021-01-26-11-22 (snowfall)")
    ap.add_argument("out", type=Path, help="output root; frames go to OUT/<sequence>_<start>")
    ap.add_argument("--start", type=int, default=3300, help="first scan (default: %(default)s)")
    ap.add_argument("--frames", type=int, default=200, help="number of scans (default: 200)")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/boreas"),
                    help="download folder (default: %(default)s)")
    ap.add_argument("--keep-raw", action="store_true", help="keep the downloaded .bin scans")
    ap.add_argument("--no-compress", action="store_true", help="write plain .npz files")
    a = ap.parse_args(argv)
    src = download(a.sequence, a.raw, a.start, a.frames)
    out = a.out / f"{a.sequence}_{a.start}"
    n = convert(src, out, a.start, a.frames, compress=not a.no_compress)
    if not a.keep_raw:
        stamps = read_poses(src / "applanix" / "lidar_poses.csv")[a.start : a.start + a.frames, 0]
        for t in stamps.astype(np.int64):
            (src / "lidar" / f"{t}.bin").unlink(missing_ok=True)
    _log(f"wrote {n} frames to {out}")


if __name__ == "__main__":
    main()
