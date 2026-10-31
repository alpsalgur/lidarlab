"""WADS (Winter Adverse Driving dataSet, Michigan Tech) -> per-frame .npz files.

WADS has point-wise labels for falling and accumulated snow, so snow filters can be scored
on real data. Its download page asks for a browser (it blocks scripts), so get a sequence
by hand from https://digitalcommons.mtu.edu/wads/ , unzip it anywhere and point this at
the folder. It reads the SemanticKITTI layout found under that folder:

- ``velodyne/NNNNNN.bin``: float32 x, y, z, intensity per point
- ``labels/NNNNNN.label``: uint32 per point, semantic class in the lower 16 bits
- ``times.txt`` (optional): seconds per scan; else 10 Hz is assumed

Output, flat layout: ``x, y, z, intensity`` float32, ``echo_number`` uint8 (1),
``label`` uint16 (the semantic class) and ``timestamp``. ``recording.json`` lists the
snow classes as ``noise_labels``; the converter prints which classes it found. WADS is
licensed CC BY 4.0 (Kurup & Bos, 2021).

    python -m lidarlab.datasets.wads path/to/unzipped/sequence data/wads/seq_11
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SNOW_LABELS = {110: "falling snow", 111: "accumulated snow"}  # WADS classes added to KITTI's


def _log(msg: str) -> None:
    print(msg, flush=True)


def find_scans(root: Path) -> Path:
    """The ``velodyne`` folder under ``root`` (``root`` itself may be it)."""
    if root.name == "velodyne" and root.is_dir():
        return root
    found = sorted(p for p in root.rglob("velodyne") if p.is_dir())
    if not found:
        raise FileNotFoundError(f"no 'velodyne' folder under {root}")
    if len(found) > 1:
        names = ", ".join(str(p.parent.relative_to(root)) for p in found)
        raise ValueError(f"several sequences under {root} ({names}); point at one of them")
    return found[0]


def convert(src: Path, out_dir: Path, noise_labels=(110,), rate_hz: float = 10.0,
            compress: bool = True, log=_log) -> dict[int, int]:
    """Convert one sequence; returns the number of points per semantic class."""
    velodyne = find_scans(src)
    labels_dir = velodyne.parent / "labels"
    scans = sorted(velodyne.glob("*.bin"))
    if not scans:
        raise FileNotFoundError(f"no .bin scans in {velodyne}")
    times_file = velodyne.parent / "times.txt"
    times = np.loadtxt(times_file, ndmin=1) if times_file.exists() else None
    out_dir.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compress else np.savez
    counts: dict[int, int] = {}
    labelled = 0
    for k, scan in enumerate(scans):
        pts = np.fromfile(scan, dtype=np.float32).reshape(-1, 4)
        arrays = {
            "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "intensity": pts[:, 3],
            "echo_number": np.ones(len(pts), np.uint8),
            "timestamp": np.float64(times[k] if times is not None and k < len(times)
                                    else k / rate_hz),
        }
        label_file = labels_dir / f"{scan.stem}.label"
        if label_file.exists():
            semantic = (np.fromfile(label_file, dtype=np.uint32) & 0xFFFF).astype(np.uint16)
            if len(semantic) != len(pts):
                raise ValueError(f"{label_file.name}: {len(semantic)} labels for {len(pts)} points")
            arrays["label"] = semantic
            values, n = np.unique(semantic, return_counts=True)
            for v, c in zip(values.tolist(), n.tolist(), strict=True):
                counts[v] = counts.get(v, 0) + c
            labelled += 1
        save(out_dir / f"frame_{k:06d}.npz", **arrays)
        if log and ((k + 1) % 50 == 0 or k + 1 == len(scans)):
            log(f"{k + 1}/{len(scans)} frames")
    info = {
        "dataset": "WADS",
        "source": str(velodyne.parent.name),
        "license": "CC BY 4.0 (Kurup & Bos, The Winter Adverse Driving dataSet, 2021)",
        "layout": "flat",
        "echoes": {"1": "single return"},
        "labels": {str(k): v for k, v in SNOW_LABELS.items()},
        "noise_labels": [int(v) for v in noise_labels],
        "units": {"xyz": "m, sensor frame (x forward, y left, z up)",
                  "timestamp": "s" + ("" if times is not None else f" (assumed {rate_hz} Hz)")},
        "frames": len(scans),
        "labelled_frames": labelled,
    }
    (out_dir / "recording.json").write_text(json.dumps(info, indent=2))
    if log:
        total = sum(counts.values()) or 1
        log(f"{labelled}/{len(scans)} scans have labels; classes (share of points):")
        for v, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            log(f"  {v:>5} {SNOW_LABELS.get(v, ''):18} {100 * c / total:6.2f} %")
        if labelled and not set(noise_labels) & set(counts):
            log(f"warning: none of the noise labels {list(noise_labels)} occur; "
                "pass the snow class with --noise-labels")
    return counts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Convert a WADS (SemanticKITTI layout) sequence.")
    ap.add_argument("src", type=Path, help="unzipped sequence (any folder above velodyne/)")
    ap.add_argument("out", type=Path, help="output folder")
    ap.add_argument("--noise-labels", type=int, nargs="+", default=[110],
                    help="classes a snow filter should remove (default: 110, falling snow)")
    ap.add_argument("--no-compress", action="store_true", help="write plain .npz files")
    a = ap.parse_args(argv)
    convert(a.src, a.out, noise_labels=a.noise_labels, compress=not a.no_compress)
    _log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
