"""KITTI raw drives -> per-frame .npz files in the flat layout.

Downloads one synchronized drive from the public KITTI raw bucket, extracts only the
Velodyne scans and the OXTS (GPS/IMU) records, and writes one ``.npz`` per scan:

- ``x, y, z, intensity``: (M,) float32; intensity is the KITTI reflectance (0-1)
- ``echo_number``: (M,) uint8, always 1. The HDL-64E reports one real return per
  pulse; echo 0 stays reserved for cover reflections, as in the target data.
- ``roll, pitch, yaw``: float32 rad from OXTS; ``timestamp``: float64 POSIX s

KITTI is licensed CC BY-NC-SA 3.0 (non-commercial use only).

    python -m lidarlab.datasets.kitti 2011_09_26_drive_0011 data/kitti
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"


def _log(msg: str) -> None:
    print(msg, flush=True)


def drive_dir(raw_root: Path, drive: str) -> Path:
    """Folder of an extracted drive: raw_root/2011_09_26/2011_09_26_drive_0011_sync."""
    return raw_root / drive[:10] / f"{drive}_sync"


def download(drive: str, dest: Path, log=_log) -> Path:
    """Download the synced zip of ``drive`` into ``dest`` (skipped if already there)."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{drive}_sync.zip"
    if zip_path.exists():
        return zip_path
    part = zip_path.with_name(zip_path.name + ".part")
    url = f"{BASE_URL}/{drive}/{drive}_sync.zip"
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done, next_log, start = 0, 0, time.perf_counter()
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if log and done >= next_log:
                speed = done / max(time.perf_counter() - start, 1e-6) / 1e6
                log(f"{done / 1e6:,.0f} / {total / 1e6:,.0f} MB ({speed:.1f} MB/s)")
                next_log += 100_000_000
    part.replace(zip_path)
    return zip_path


def extract(zip_path: Path, raw_root: Path) -> None:
    """Extract only the Velodyne scans and OXTS records; camera images are skipped."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if "/velodyne_points/" in m or "/oxts/" in m]
        zf.extractall(raw_root, members)


def parse_timestamp(line: str) -> float:
    """'2011-09-26 13:02:25.964389445' -> POSIX seconds (KITTI stamps have 9 digits)."""
    whole, _, frac = line.strip().partition(".")
    dt = datetime.strptime(whole, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    return dt.timestamp() + float("0." + (frac or "0"))


def convert(src: Path, out_dir: Path, compress: bool = True, log=_log) -> int:
    """Convert an extracted drive folder; returns the number of frames written."""
    velo = {p.stem: p for p in (src / "velodyne_points" / "data").glob("*.bin")}
    oxts = {p.stem: p for p in (src / "oxts" / "data").glob("*.txt")}
    lines = (src / "velodyne_points" / "timestamps.txt").read_text().splitlines()
    stamps = [parse_timestamp(s) for s in lines if s.strip()]
    stems = sorted(velo.keys() & oxts.keys())
    if not stems:
        raise FileNotFoundError(f"no Velodyne/OXTS pairs under {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compress else np.savez
    for k, stem in enumerate(stems):
        pts = np.fromfile(velo[stem], dtype=np.float32).reshape(-1, 4)
        roll, pitch, yaw = np.loadtxt(oxts[stem])[3:6]
        save(
            out_dir / f"frame_{k:06d}.npz",
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            intensity=pts[:, 3],
            echo_number=np.ones(len(pts), np.uint8),
            roll=np.float32(roll),
            pitch=np.float32(pitch),
            yaw=np.float32(yaw),
            timestamp=np.float64(stamps[int(stem)]),
        )
        if log and ((k + 1) % 50 == 0 or k + 1 == len(stems)):
            log(f"{k + 1}/{len(stems)} frames")
    info = {
        "dataset": "KITTI raw",
        "source": f"KITTI raw {src.name}",
        "license": "CC BY-NC-SA 3.0",
        "sensor": "Velodyne HDL-64E, 10 Hz",
        "layout": "flat",
        "echoes": {"1": "single real return"},
        "units": {
            "xyz": "m, sensor frame (x forward, y left, z up)",
            "roll/pitch/yaw": "rad, OXTS",
            "timestamp": "s (POSIX)",
            "intensity": "reflectance 0-1",
        },
        "frames": len(stems),
    }
    (out_dir / "recording.json").write_text(json.dumps(info, indent=2))
    return len(stems)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download a KITTI raw drive and convert it to .npz.")
    ap.add_argument("drive", help="drive name, e.g. 2011_09_26_drive_0011")
    ap.add_argument("out", type=Path, help="output root; frames go to OUT/<drive>")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/kitti"),
                    help="download and extraction folder (default: %(default)s)")
    ap.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip")
    ap.add_argument("--no-compress", action="store_true", help="write plain .npz files")
    a = ap.parse_args(argv)
    src = drive_dir(a.raw, a.drive)
    if not (src / "velodyne_points").is_dir():
        zip_path = download(a.drive, a.raw)
        _log("extracting Velodyne scans and OXTS records ...")
        extract(zip_path, a.raw)
        if not a.keep_zip:
            zip_path.unlink()
    n = convert(src, a.out / a.drive, compress=not a.no_compress)
    _log(f"wrote {n} frames to {a.out / a.drive}")


if __name__ == "__main__":
    main()
