import json

import numpy as np

from lidarlab.datasets import synthetic

BEAMS, AZIMUTH = 32, 512


def small(**overrides) -> synthetic.Config:
    return synthetic.Config(frames=3, n_beams=BEAMS, n_azimuth=AZIMUTH, **overrides)


def test_echo_axis_layout(tmp_path):
    synthetic.generate(tmp_path, small(), log=None)
    files = sorted(tmp_path.glob("frame_*.npz"))
    assert len(files) == 3
    assert (tmp_path / "recording.json").exists()
    with np.load(files[0]) as f:
        for key in ("x", "y", "z", "intensity"):
            assert f[key].shape == (3, BEAMS * AZIMUTH)
        assert f["ring"].shape == (BEAMS * AZIMUTH,)
        for key in ("roll", "pitch", "yaw", "timestamp"):
            assert f[key].shape == ()
        assert np.array_equal(np.isnan(f["x"]), np.isnan(f["intensity"]))
        cover_range = np.sqrt(f["x"][0] ** 2 + f["y"][0] ** 2 + f["z"][0] ** 2)
        assert np.nanmax(cover_range) < 0.1  # echo 0 sits on the sensor cover
        assert np.isfinite(f["x"][1]).mean() > 0.5  # most pulses have a first real return


def test_flat_layout_holds_the_same_points(tmp_path):
    synthetic.generate(tmp_path / "a", small(), log=None)
    synthetic.generate(tmp_path / "b", small(layout="flat"), log=None)
    with np.load(tmp_path / "a" / "frame_000002.npz") as a, \
            np.load(tmp_path / "b" / "frame_000002.npz") as b:
        valid = np.isfinite(a["x"])
        assert np.array_equal(b["x"], a["x"][valid])
        assert np.array_equal(b["intensity"], a["intensity"][valid])
        assert np.array_equal(b["echo_number"], np.nonzero(valid)[0])
        assert np.array_equal(b["ring"], np.broadcast_to(a["ring"], valid.shape)[valid])


def test_rain_adds_cover_and_second_echoes(tmp_path):
    synthetic.generate(tmp_path / "dry", small(rain=0.0), log=None)
    synthetic.generate(tmp_path / "wet", small(rain=1.0), log=None)
    # frame 2 of 3 is at peak rain
    with np.load(tmp_path / "dry" / "frame_000002.npz") as dry, \
            np.load(tmp_path / "wet" / "frame_000002.npz") as wet:
        n_dry = np.isfinite(dry["x"]).sum(axis=1)
        n_wet = np.isfinite(wet["x"]).sum(axis=1)
    assert n_wet[0] > n_dry[0]  # wet cover
    assert n_wet[2] > n_dry[2]  # drops and spray push real targets to echo 2


def test_workers_give_the_same_frames(tmp_path):
    synthetic.generate(tmp_path / "serial", small(), log=None)
    synthetic.generate(tmp_path / "parallel", small(), workers=2, log=None)
    for k in range(3):
        name = f"frame_{k:06d}.npz"
        with np.load(tmp_path / "serial" / name) as a, np.load(tmp_path / "parallel" / name) as b:
            for key in a.files:
                assert np.array_equal(a[key], b[key], equal_nan=True), (name, key)


def test_labels_mark_cover_rain_and_spray(tmp_path):
    synthetic.generate(tmp_path / "dry", small(rain=0.0), log=None)
    synthetic.generate(tmp_path / "wet", small(rain=1.0), log=None)
    with np.load(tmp_path / "dry" / "frame_000002.npz") as dry, \
            np.load(tmp_path / "wet" / "frame_000002.npz") as wet:
        for f in (dry, wet):
            valid = np.isfinite(f["x"])
            assert f["label"].shape == f["x"].shape
            assert np.all(f["label"][0][valid[0]] == 0)  # echo 0 is the cover
            assert np.all(f["label"][1:][valid[1:]] >= 1)
            ground = f["label"] == 1
            assert np.all(np.abs(f["z"][ground]) < 3.0)  # ground below a 1.9 m high sensor
        dry_labels = set(np.unique(dry["label"][np.isfinite(dry["x"])]).tolist())
        wet_labels = set(np.unique(wet["label"][np.isfinite(wet["x"])]).tolist())
    assert not dry_labels & {5, 6}
    assert {5, 6} <= wet_labels
    info = json.loads((tmp_path / "wet" / "recording.json").read_text())
    assert info["labels"]["5"] == "rain" and info["noise_labels"] == [0, 5, 6]
