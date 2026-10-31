"""A test run: a folder of per-frame ``.npz`` files, loaded in the background and cached."""

from __future__ import annotations

import json
import operator
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from .frame import Frame
from .npz import read_frame

BACKGROUND = 1_000_000  # queue priority offset of background requests


def natural_key(path: Path) -> list:
    """Sort key that puts frame_2 before frame_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def default_cache_bytes() -> int:
    """A quarter of the RAM, at most 4 GiB."""
    try:
        import psutil

        total = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - psutil missing or unsupported here
        total = 8 << 30
    return int(min(total // 4, 4 << 30))


def _nbytes(frame: Frame) -> int:
    return frame.xyz.nbytes + frame.index.nbytes + sum(a.nbytes for a in frame.fields.values())


def _freeze(frame: Frame) -> Frame:
    for a in (frame.xyz, frame.index, *frame.fields.values()):
        a.flags.writeable = False
    return frame


class Recording:
    """A test run: one ``.npz`` per frame in a folder, in natural name order.

    ``rec[i]`` returns frame ``i`` (negative indices count from the end) and reads it if
    needed. Loaded frames stay in an LRU cache bounded by ``cache_budget`` bytes; their
    arrays are read-only. ``prefetch`` queues frames for the loader threads, which call
    ``on_loaded(i)`` after frame ``i`` finished loading or failed (see ``errors``).
    """

    def __init__(self, path, cache_bytes: int | None = None, workers: int = 3):
        self.path = Path(path)
        if not self.path.is_dir():
            raise NotADirectoryError(f"not a folder: {self.path}")
        self.files = sorted(self.path.glob("*.npz"), key=natural_key)
        if not self.files:
            raise FileNotFoundError(f"no .npz files in {self.path}")
        info = self.path / "recording.json"
        self.info: dict = json.loads(info.read_text()) if info.exists() else {}
        self.cache_budget = cache_bytes or default_cache_bytes()
        self.on_loaded: Callable[[int], None] | None = None
        self.errors: dict[int, Exception] = {}
        self.load_ms: dict[int, float] = {}
        self.cache_version = 0  # bumped whenever the set of cached frames changes
        self._cache: OrderedDict[int, Frame] = OrderedDict()
        self._sizes: dict[int, int] = {}
        self._cached_bytes = 0
        self._queue: dict[int, int] = {}  # frame -> priority, lowest loads first
        self._busy: set[int] = set()
        self._cv = threading.Condition()
        self._closed = False
        self._rate: float | None = None
        for k in range(workers):
            threading.Thread(target=self._worker, name=f"lidarlab-load-{k}", daemon=True).start()

    @property
    def name(self) -> str:
        return self.path.name

    def __len__(self) -> int:
        return len(self.files)

    def __repr__(self) -> str:
        return f"Recording({self.name!r}: {len(self)} frames, {len(self._cache)} cached)"

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self.get(j) for j in range(*i.indices(len(self)))]
        return self.get(i)

    def __iter__(self):
        return (self.get(i) for i in range(len(self)))

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _check(self, i) -> int:
        i = operator.index(i)
        n = len(self.files)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"frame {i} out of range 0-{n - 1}")
        return i

    # access

    def get(self, i) -> Frame:
        """Frame ``i``; waits for a loader thread already reading it, else reads it here."""
        i = self._check(i)
        with self._cv:
            while True:
                frame = self._cache.get(i)
                if frame is not None:
                    self._cache.move_to_end(i)
                    return frame.view()
                if i not in self._busy:
                    break
                self._cv.wait()
            self._queue.pop(i, None)
            self._busy.add(i)
        return self._load(i, raise_errors=True).view()

    def peek(self, i) -> Frame | None:
        """Frame ``i`` if it is cached, else None; never blocks on loading."""
        with self._cv:
            frame = self._cache.get(i)
            if frame is None:
                return None
            self._cache.move_to_end(i)
            return frame.view()

    def is_cached(self, i) -> bool:
        with self._cv:
            return i in self._cache

    def cached_mask(self) -> np.ndarray:
        """Boolean array: which frames are in the cache."""
        mask = np.zeros(len(self), bool)
        with self._cv:
            mask[list(self._cache)] = True
        return mask

    @property
    def cached_bytes(self) -> int:
        return self._cached_bytes

    def timestamp(self, i) -> float | None:
        """POSIX time of frame ``i``; reads only that key if the frame is not cached."""
        i = self._check(i)
        frame = self.peek(i)
        if frame is not None:
            t = frame.meta.get("timestamp")
        else:
            with np.load(self.files[i]) as f:
                t = f["timestamp"][()] if "timestamp" in f.files else None
        return None if t is None else float(t)

    @property
    def rate_hz(self) -> float:
        """Frame rate from the first timestamps, else recording.json, else 10 Hz."""
        if self._rate is None:
            stamps = [self.timestamp(i) for i in range(min(6, len(self)))]
            stamps = [t for t in stamps if t is not None]
            dt = float(np.median(np.diff(stamps))) if len(stamps) >= 2 else 0.0
            config_rate = self.info.get("config", {}).get("rate_hz")
            self._rate = 1.0 / dt if dt > 0 else float(config_rate or 10.0)
        return self._rate

    # background loading

    def prefetch(self, indices: Iterable[int], background: bool = False) -> None:
        """Queue frames for the loader threads, first ones first.

        A foreground call replaces the previous foreground queue, so frames the playhead has
        left behind are dropped. Background requests wait behind it until they are loaded or
        the cache is full.
        """
        n = len(self.files)
        offset = BACKGROUND if background else 0
        with self._cv:
            if not background:
                for j in [j for j, p in self._queue.items() if p < BACKGROUND]:
                    del self._queue[j]
            for rank, j in enumerate(indices):
                j = int(j)
                if 0 <= j < n and j not in self._cache and j not in self._busy:
                    self._queue[j] = min(self._queue.get(j, BACKGROUND * 2), offset + rank)
            self._cv.notify_all()

    def cache_all(self, start: int = 0) -> None:
        """Queue every frame from ``start`` on (wrapping around) as background work."""
        n = len(self)
        self.prefetch(((start + k) % n for k in range(n)), background=True)

    def close(self) -> None:
        """Stop the loader threads; cached frames stay readable."""
        with self._cv:
            self._closed = True
            self._queue.clear()
            self._cv.notify_all()

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                i = min(self._queue, key=self._queue.__getitem__)
                priority = self._queue.pop(i)
                if i in self._cache or i in self._busy:
                    continue
                if priority >= BACKGROUND and self._full():
                    # background frames would only evict each other from now on
                    for j in [j for j, p in self._queue.items() if p >= BACKGROUND]:
                        del self._queue[j]
                    continue
                self._busy.add(i)
            self._load(i, raise_errors=False)

    def _full(self) -> bool:
        """True if one more frame of average size would exceed the budget (lock held)."""
        if not self._cache:
            return False
        return self._cached_bytes * (len(self._cache) + 1) / len(self._cache) > self.cache_budget

    def _load(self, i: int, raise_errors: bool) -> Frame | None:
        """Read frame ``i``, which the caller has marked busy."""
        start = time.perf_counter()
        frame = error = None
        try:
            frame = _freeze(read_frame(self.files[i], frame_no=i))
        except Exception as e:  # noqa: BLE001 - kept in self.errors, re-raised for get()
            error = e
        with self._cv:
            self._busy.discard(i)
            if frame is not None:
                self.load_ms[i] = (time.perf_counter() - start) * 1e3
                self.errors.pop(i, None)
                if i in self._cache:
                    frame = self._cache[i]
                else:
                    self._cache[i] = frame
                    self._sizes[i] = _nbytes(frame)
                    self._cached_bytes += self._sizes[i]
                    self._evict(keep=i)
                    self.cache_version += 1
            else:
                self.errors[i] = error
            self._cv.notify_all()
        if self.on_loaded is not None:
            self.on_loaded(i)
        if error is not None and raise_errors:
            raise error
        return frame

    def _evict(self, keep: int) -> None:
        """Drop least recently used frames until the cache fits its budget (lock held)."""
        while self._cached_bytes > self.cache_budget and len(self._cache) > 1:
            j = next(iter(self._cache))
            if j == keep:
                self._cache.move_to_end(j)
                continue
            del self._cache[j]
            self._cached_bytes -= self._sizes.pop(j)
