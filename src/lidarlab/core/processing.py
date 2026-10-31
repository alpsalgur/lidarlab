"""Run the pipeline on the frames ahead of the playhead in background threads and keep the
results, so heavy filters neither hold up playback nor freeze the window."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable

from .pipeline import Pipeline, Processed
from .recording import Recording, _nbytes


class Processor:
    """Pipeline outputs per frame, computed in the background and cached.

    ``request`` lists the frames wanted next; workers take the first one that is loaded.
    Results belong to one pipeline version: any change to the steps or their parameters
    makes older results invisible (``get`` returns None) and they are dropped on
    ``invalidate``. ``on_done(i)`` is called from a worker thread when frame ``i`` is ready.
    """

    def __init__(self, pipeline: Pipeline, cache_bytes: int = 1 << 30, workers: int = 2):
        self.pipeline = pipeline
        self.cache_budget = cache_bytes
        self.on_done: Callable[[int], None] | None = None
        self.rec: Recording | None = None
        self.version = 0  # bumped whenever the set of cached results changes
        self._cache: OrderedDict[int, Processed] = OrderedDict()
        self._sizes: dict[int, int] = {}
        self._bytes = 0
        self._queue: list[int] = []
        self._busy: set[int] = set()
        self._cv = threading.Condition()
        self._closed = False
        for k in range(workers):
            threading.Thread(target=self._worker, name=f"lidarlab-process-{k}", daemon=True).start()

    def set_recording(self, rec: Recording | None) -> None:
        with self._cv:
            self.rec = rec
            self._clear()

    def invalidate(self) -> None:
        """Drop all results and requests (the pipeline changed)."""
        with self._cv:
            self._clear()

    def _clear(self) -> None:
        self._cache.clear()
        self._sizes.clear()
        self._bytes = 0
        self._queue.clear()
        self.version += 1

    # results

    def get(self, i: int) -> Processed | None:
        """Result for frame ``i`` from the current pipeline version, else None."""
        with self._cv:
            result = self._cache.get(i)
            if result is None or result.version != self.pipeline.version:
                return None
            self._cache.move_to_end(i)
            return result

    def has(self, i: int) -> bool:
        return self.get(i) is not None

    def cached(self) -> list[int]:
        with self._cv:
            version = self.pipeline.version
            return [i for i, r in self._cache.items() if r.version == version]

    @property
    def cached_bytes(self) -> int:
        return self._bytes

    def compute(self, i: int, raw) -> Processed:
        """Run the pipeline on ``raw`` (frame ``i``) here and keep the result."""
        result = self.pipeline.process(raw)
        with self._cv:
            self._store(i, result)
        return result

    def request(self, indices: Iterable[int]) -> None:
        """Frames to compute next, most wanted first (replaces the previous request)."""
        if not self.pipeline.active:
            return
        with self._cv:
            version = self.pipeline.version
            self._queue = [i for i in indices
                           if i not in self._busy and not self._fresh(i, version)]
            self._cv.notify_all()

    def notify(self) -> None:
        """A frame finished loading: wake workers waiting for one."""
        with self._cv:
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._queue.clear()
            self._cv.notify_all()

    # internals (lock held unless noted)

    def _fresh(self, i: int, version: int) -> bool:
        result = self._cache.get(i)
        return result is not None and result.version == version

    def _store(self, i: int, result: Processed) -> bool:
        if result.version != self.pipeline.version:
            return False  # the pipeline changed while this frame was processed
        if i in self._cache:
            self._bytes -= self._sizes.pop(i)
        self._cache[i] = result
        self._sizes[i] = _nbytes(result.frame)
        self._bytes += self._sizes[i]
        while self._bytes > self.cache_budget and len(self._cache) > 1:
            j = next(iter(self._cache))
            if j == i:
                self._cache.move_to_end(j)
                continue
            del self._cache[j]
            self._bytes -= self._sizes.pop(j)
        self.version += 1
        return True

    def _next(self) -> int | None:
        rec = self.rec
        for i in self._queue:
            if rec is not None and i not in self._busy and rec.is_cached(i):
                return i
        return None

    def _worker(self) -> None:  # lock not held while processing
        while True:
            with self._cv:
                while not self._closed and (i := self._next()) is None:
                    self._cv.wait(timeout=0.25)
                if self._closed:
                    return
                self._queue.remove(i)
                self._busy.add(i)
                rec = self.rec
            raw = rec.peek(i)
            result = self.pipeline.process(raw) if raw is not None else None
            with self._cv:
                self._busy.discard(i)
                stored = result is not None and rec is self.rec and self._store(i, result)
            if stored and self.on_done is not None:
                self.on_done(i)
