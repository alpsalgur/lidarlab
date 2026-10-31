"""Per-frame numbers for a whole test run: point counts, what the pipeline removes, echo
shares, intensity, attitude, ground-truth noise and the points inside a selection.

    store = MetricStore(len(rec))
    compute_metrics(rec, pipe, store)          # all frames, in threads
    store.values["removed_pct"]                # one value per frame, NaN where unknown
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np

from .batch import default_workers
from .frame import Frame
from .npz import read_frame
from .pipeline import Pipeline

# key: (label, unit); the order is the order of the metric menu
METRICS = {
    "points": ("points loaded", ""),
    "kept": ("points after the pipeline", ""),
    "removed_pct": ("removed by the pipeline", "%"),
    "cover_pct": ("echo 0 (cover) share", "%"),
    "second_pct": ("echo 2 share", "%"),
    "intensity": ("mean intensity after the pipeline", ""),
    "roll": ("roll", "°"),
    "pitch": ("pitch", "°"),
    "noise_pct": ("labelled noise", "%"),
    "noise_left_pct": ("labelled noise left after the pipeline", "%"),
    "selected": ("points in the selection", ""),
}
PIPELINE_METRICS = ("kept", "removed_pct", "intensity", "noise_left_pct", "selected")


def frame_metrics(raw: Frame, out: Frame, noise_labels: Iterable[int] = (),
                  select: Callable[[np.ndarray], np.ndarray] | None = None) -> dict[str, float]:
    """The metrics of one frame; ``out`` is ``raw`` after the pipeline, ``select(xyz)`` a
    selection mask (counted on ``out``)."""
    n = max(len(raw), 1)
    m = {"points": float(len(raw)), "kept": float(len(out)),
         "removed_pct": 100.0 * (len(raw) - len(out)) / n}
    if "echo" in raw.fields:
        echo = raw.fields["echo"]
        m["cover_pct"] = 100.0 * np.count_nonzero(echo == 0) / n
        m["second_pct"] = 100.0 * np.count_nonzero(echo == 2) / n
    if "intensity" in out.fields and len(out):
        m["intensity"] = float(np.mean(out.fields["intensity"]))
    for key in ("roll", "pitch"):
        if key in raw.meta:
            m[key] = float(np.degrees(float(raw.meta[key])))
    noise_labels = list(noise_labels)
    if noise_labels and "label" in raw.fields:
        noise = np.count_nonzero(np.isin(raw.fields["label"], noise_labels))
        m["noise_pct"] = 100.0 * noise / n
        if noise and "label" in out.fields:
            left = np.count_nonzero(np.isin(out.fields["label"], noise_labels))
            m["noise_left_pct"] = 100.0 * left / noise
    if select is not None:
        m["selected"] = float(np.count_nonzero(select(out.xyz)))
    return m


class MetricStore:
    """One value per frame and metric, NaN until known. Thread-safe; ``version`` changes
    with every update so views know when to redraw."""

    def __init__(self, n_frames: int):
        self.n = n_frames
        self.values = {key: np.full(n_frames, np.nan) for key in METRICS}
        self.version = 0
        self._lock = threading.Lock()

    def put(self, i: int, metrics: dict[str, float]) -> None:
        with self._lock:
            for key, value in metrics.items():
                if key in self.values:
                    self.values[key][i] = value
            self.version += 1

    def clear(self, keys: Iterable[str] | None = None) -> None:
        with self._lock:
            for key in (self.values if keys is None else keys):
                self.values[key][:] = np.nan
            self.version += 1

    def known(self, key: str) -> int:
        return int(np.count_nonzero(np.isfinite(self.values[key])))


def compute_metrics(source, pipeline: Pipeline | None, store: MetricStore, frames=None,
                    workers: int | None = None, noise_labels: Iterable[int] = (),
                    select=None, progress=None) -> int:
    """Fill ``store`` for ``frames`` (default all) of ``source`` (a Recording or a list of
    files): read, run ``pipeline``, measure, in threads. ``progress(done, total)`` returning
    False cancels. Returns the number of frames measured."""
    files = list(getattr(source, "files", source))
    indices = list(range(len(files))) if frames is None else list(frames)
    pipe = pipeline.snapshot() if pipeline is not None else Pipeline()
    noise_labels = list(noise_labels)
    workers = workers or default_workers()

    def measure(i: int) -> None:
        raw = read_frame(files[i], frame_no=i)
        out = pipe.process(raw).frame
        store.put(i, frame_metrics(raw, out, noise_labels, select))

    done = 0
    jobs = iter(indices)
    pending = set()
    with ThreadPoolExecutor(workers, thread_name_prefix="lidarlab-metrics") as pool:
        while True:
            while len(pending) < 2 * workers and (i := next(jobs, None)) is not None:
                pending.add(pool.submit(measure, i))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
                done += 1
            if progress is not None and progress(done, len(indices)) is False:
                for future in pending:
                    future.cancel()
                break
    return done
