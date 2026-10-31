import time

import numpy as np
import pytest

import lidarlab as lab
from lidarlab import Frame, Pipeline, Processor
from lidarlab.core import filters as F


def wait(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            return False
        time.sleep(0.01)
    return True


@pytest.fixture
def rec(tmp_path):
    rng = np.random.default_rng(1)
    for k in range(6):
        xyz = rng.normal(size=(500, 3)).astype(np.float32)
        lab.write_frame(Frame(xyz, {"intensity": rng.uniform(0, 100, 500).astype(np.float32)},
                              {"timestamp": 100.0 + 0.1 * k}), tmp_path / f"frame_{k:06d}.npz")
    r = lab.open(tmp_path)
    yield r
    r.close()


def test_background_results_follow_the_pipeline_version(rec):
    p = Pipeline()
    p.add(F.Threshold(field="intensity", min=0, max=50))
    proc = Processor(p, workers=2)
    proc.set_recording(rec)
    done = []
    proc.on_done = done.append
    rec.prefetch(range(6))
    proc.request(range(6))
    assert wait(lambda: len(proc.cached()) == 6)
    assert sorted(done) == list(range(6))
    out = proc.get(3)
    raw = rec[3]
    assert len(out.frame) == np.count_nonzero(raw.intensity <= 50)
    assert out.stats["threshold"].n_in == 500 and out.stats["threshold"].ms is not None

    p.set("threshold", max=10)  # new version: old results are stale
    assert proc.get(3) is None
    proc.invalidate()
    assert proc.cached() == []
    result = proc.compute(3, raw)
    assert len(result.frame) == np.count_nonzero(raw.intensity <= 10)
    assert proc.get(3) is result
    proc.close()


def test_waits_for_frames_to_load_and_respects_the_budget(rec):
    p = Pipeline()
    p.add(lambda f: f.intensity > 20, name="bright")
    proc = Processor(p, cache_bytes=1, workers=1)  # room for a single result
    proc.set_recording(rec)
    proc.request([4, 5])
    time.sleep(0.3)
    assert proc.cached() == []  # not loaded yet: nothing to process
    rec.prefetch([4, 5])
    proc.notify()
    assert wait(lambda: proc.cached() == [5] or proc.cached() == [4])
    assert wait(lambda: proc.version >= 3)  # both done; only the last one fits
    assert len(proc.cached()) == 1
    proc.close()


def test_nothing_to_do_without_active_steps(rec):
    p = Pipeline()
    proc = Processor(p, workers=1)
    proc.set_recording(rec)
    rec.prefetch(range(6))
    proc.request(range(6))
    time.sleep(0.3)
    assert proc.cached() == []
    proc.close()


def test_failing_steps_are_reported_in_the_stats(rec):
    p = Pipeline()
    p.add(lambda f: 1 / 0, name="broken")
    proc = Processor(p, workers=1)
    proc.set_recording(rec)
    result = proc.compute(0, rec[0])
    assert isinstance(result.stats["broken"].error, ZeroDivisionError)
    assert len(result.frame) == 500
    p.record(result.stats)
    assert isinstance(p["broken"].error, ZeroDivisionError)
    proc.close()
