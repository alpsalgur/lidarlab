import numpy as np
import pytest

import lidarlab as lab
from lidarlab import Frame, Pipeline
from lidarlab.core import filters as F
from lidarlab.core.metrics import METRICS, MetricStore, compute_metrics, frame_metrics
from lidarlab.datasets import synthetic


def test_frame_metrics():
    raw = Frame(np.c_[np.arange(10.0), np.zeros(10), np.zeros(10)],
                {"echo": np.array([0, 0, 1, 1, 1, 1, 2, 2, 1, 1], np.uint8),
                 "intensity": np.arange(10, dtype=np.float32),
                 "label": np.array([0, 0, 1, 5, 5, 1, 6, 1, 1, 1], np.uint8)},
                {"roll": np.radians(1.5), "pitch": 0.0})
    out = raw[raw.echo != 0][1:]  # drop the cover echo and point 2 (ground)
    m = frame_metrics(raw, out, noise_labels=[0, 5, 6], select=lambda xyz: xyz[:, 0] > 7)
    assert m["points"] == 10 and m["kept"] == 7 and m["removed_pct"] == pytest.approx(30)
    assert m["cover_pct"] == 20 and m["second_pct"] == 20
    assert m["intensity"] == pytest.approx(np.mean(np.arange(3, 10)))
    assert m["roll"] == pytest.approx(1.5) and m["pitch"] == 0
    assert m["noise_pct"] == pytest.approx(50)
    assert m["noise_left_pct"] == pytest.approx(60)  # rain 3, 4 and spray 6: 3 of 5 noise points
    assert m["selected"] == 2
    assert set(m) <= set(METRICS)


def test_compute_metrics_fills_the_store(tmp_path):
    synthetic.generate(tmp_path, synthetic.Config(frames=4, n_beams=32, n_azimuth=512), log=None)
    rec = lab.open(tmp_path)
    pipe = Pipeline()
    pipe.add(F.EchoSelect())
    store = MetricStore(len(rec))
    seen = []
    done = compute_metrics(rec, pipe, store, frames=[0, 2, 3], workers=2,
                           noise_labels=[0, 5, 6], progress=lambda d, t: seen.append((d, t)))
    assert done == 3 and seen[-1] == (3, 3)
    assert np.isnan(store.values["points"][1]) and store.known("points") == 3
    raw = rec[2]
    assert store.values["points"][2] == len(raw)
    assert store.values["kept"][2] == np.count_nonzero(raw.echo != 0)
    store.clear(["kept"])
    assert store.known("kept") == 0 and store.known("points") == 3
    cancelled = MetricStore(len(rec))
    assert compute_metrics(rec, pipe, cancelled, workers=1, progress=lambda d, t: False) < 4
