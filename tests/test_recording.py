import threading
import time

import numpy as np
import pytest

from lidarlab import Recording
from lidarlab.core.recording import BACKGROUND

POINTS = 1000
FRAME_BYTES = POINTS * (12 + 4 + 4)  # xyz + intensity + int32 index


def write_run(folder, n=12):
    folder.mkdir()
    rng = np.random.default_rng(0)
    for k in range(n):
        xyz = rng.normal(size=(3, POINTS)).astype(np.float32)
        np.savez(folder / f"frame_{k}.npz", x=xyz[0], y=xyz[1], z=xyz[2],
                 intensity=rng.random(POINTS, np.float32), timestamp=np.float64(100 + 0.1 * k))
    return folder


def wait_idle(rec, timeout=10.0):
    end = time.monotonic() + timeout
    while rec._queue or rec._busy:
        assert time.monotonic() < end, "loader threads did not finish"
        time.sleep(0.01)


def test_natural_order_and_access(tmp_path):
    with Recording(write_run(tmp_path / "run")) as rec:
        assert [p.name for p in rec.files[:3]] == ["frame_0.npz", "frame_1.npz", "frame_2.npz"]
        assert rec.files[10].name == "frame_10.npz"
        assert rec[-1].meta["frame"] == 11
        assert rec.rate_hz == pytest.approx(10.0)
        assert rec.timestamp(4) == pytest.approx(100.4)
        with pytest.raises(IndexError):
            rec[12]


def test_cached_frames_share_read_only_arrays(tmp_path):
    with Recording(write_run(tmp_path / "run")) as rec:
        a, b = rec[3], rec[3]
        assert a is not b and a.xyz is b.xyz
        with pytest.raises(ValueError):
            a.xyz[0, 0] = 1
        a["intensity"] = a.intensity + 1  # replaces the array in `a` only
        assert not np.array_equal(a.intensity, b.intensity)


def test_cache_budget_evicts_least_recently_used(tmp_path):
    with Recording(write_run(tmp_path / "run"), cache_bytes=int(3.5 * FRAME_BYTES)) as rec:
        for i in range(6):
            rec[i]
        assert np.flatnonzero(rec.cached_mask()).tolist() == [3, 4, 5]
        rec[3]  # most recently used again
        rec[6]
        assert np.flatnonzero(rec.cached_mask()).tolist() == [3, 5, 6]
        assert rec.cached_bytes <= 3.5 * FRAME_BYTES


def test_prefetch_loads_in_background_and_reports(tmp_path):
    loaded, done = [], threading.Event()

    def on_loaded(i):
        loaded.append(i)
        if len(loaded) == 5:
            done.set()

    with Recording(write_run(tmp_path / "run")) as rec:
        rec.on_loaded = on_loaded
        rec.prefetch(range(5))
        assert done.wait(10)
        assert sorted(loaded) == [0, 1, 2, 3, 4]
        assert rec.cached_mask()[:5].all() and rec.peek(2) is not None and rec.peek(9) is None


def test_foreground_prefetch_replaces_the_previous_window(tmp_path):
    with Recording(write_run(tmp_path / "run"), workers=0) as rec:
        rec.prefetch([1, 2, 3])
        rec.prefetch([7, 8])
        assert rec._queue == {7: 0, 8: 1}
        rec.prefetch([4, 7], background=True)
        assert rec._queue == {7: 0, 8: 1, 4: BACKGROUND}
        rec.prefetch([9])
        assert rec._queue == {9: 0, 4: BACKGROUND}


def test_cache_all_stops_when_the_cache_is_full(tmp_path):
    budget = int(3.5 * FRAME_BYTES)
    with Recording(write_run(tmp_path / "run"), cache_bytes=budget) as rec:
        rec.cache_all()
        wait_idle(rec)
        assert len(rec.load_ms) < 12
        assert rec.cached_bytes <= budget


def test_load_errors_are_reported(tmp_path):
    folder = write_run(tmp_path / "run")
    (folder / "frame_5.npz").write_bytes(b"not a zip file")
    reported = threading.Event()
    with Recording(folder) as rec:
        with pytest.raises(Exception):  # noqa: B017 - whatever np.load raises for a broken file
            rec[5]
        assert 5 in rec.errors
        rec.on_loaded = lambda i: i == 5 and reported.set()
        rec.prefetch([5])
        assert reported.wait(10)
