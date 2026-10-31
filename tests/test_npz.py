import numpy as np
import pytest

from lidarlab import read_frame, write_frame
from lidarlab.core.npz import frame_from_arrays
from lidarlab.datasets import synthetic

NAN = np.nan


def echo_axis_arrays() -> dict:
    """3 echoes x 4 pulses; NaN = no return."""
    x = np.array([[0.05, NAN, NAN, 0.06], [10, 20, NAN, 40], [NAN, 25, NAN, NAN]], np.float32)
    like = np.where(np.isnan(x), NAN, 1.0).astype(np.float32)
    return {
        "x": x,
        "y": like,
        "z": like * -1.5,
        "intensity": x * 2,
        "ring": np.array([0, 1, 2, 3], np.uint8),
        "roll": np.float32(0.01),
        "pitch": np.float32(-0.02),
        "yaw": np.float32(1.5),
        "timestamp": np.float64(1e9 + 0.1),
    }


def test_echo_axis_read():
    f = frame_from_arrays(echo_axis_arrays())
    assert len(f) == 6
    assert f.meta["layout"] == "echo-axis" and f.meta["shape"] == (3, 4)
    np.testing.assert_array_equal(f.echo, [0, 0, 1, 1, 1, 2])
    np.testing.assert_array_equal(f.ring, [0, 3, 0, 1, 3, 1])
    np.testing.assert_array_equal(f.pulse, [0, 3, 0, 1, 3, 1])
    np.testing.assert_allclose(f.x, [0.05, 0.06, 10, 20, 40, 25])
    assert f.meta["yaw"] == np.float32(1.5) and f.meta["timestamp"] == 1e9 + 0.1
    assert list(f.fields)[:2] == ["intensity", "echo"]


def test_echo_axis_roundtrip_writes_nan_for_removed_points(tmp_path):
    src = echo_axis_arrays()
    np.savez_compressed(tmp_path / "a.npz", **src)
    f = read_frame(tmp_path / "a.npz", frame_no=3)
    assert f.meta["frame"] == 3
    write_frame(f[f.echo != 0], tmp_path / "b.npz")
    with np.load(tmp_path / "b.npz") as b:
        assert sorted(b.files) == sorted(src)
        expected = src["x"].copy()
        expected[0] = NAN
        np.testing.assert_array_equal(b["x"], expected)
        np.testing.assert_array_equal(b["intensity"][1:], src["intensity"][1:])
        assert np.isnan(b["intensity"][0]).all()
        np.testing.assert_array_equal(b["ring"], src["ring"])
        assert b["yaw"].dtype == np.float32 and b["yaw"] == src["yaw"]


def test_flat_roundtrip_keeps_key_names_and_dtypes(tmp_path):
    arrays = {
        "x": np.array([1, NAN, 3], np.float32),
        "y": np.zeros(3, np.float32),
        "z": np.zeros(3, np.float32),
        "intensity": np.array([5, 6, 7], np.float32),
        "echo_number": np.array([1, 2, 1], np.int64),
        "timestamp": np.float64(2.0),
    }
    np.savez(tmp_path / "a.npz", **arrays)
    f = read_frame(tmp_path / "a.npz")
    assert f.meta["layout"] == "flat" and len(f) == 2  # the point without coordinates is dropped
    assert f.echo.dtype == np.uint8
    np.testing.assert_array_equal(f.echo, [1, 1])
    np.testing.assert_array_equal(f.index, [0, 2])
    write_frame(f, tmp_path / "b.npz")
    with np.load(tmp_path / "b.npz") as b:
        assert sorted(b.files) == ["echo_number", "intensity", "timestamp", "x", "y", "z"]
        assert b["echo_number"].dtype == np.int64
        np.testing.assert_array_equal(b["x"], [1, 3])


def test_echo_axis_frame_written_flat_reads_back_the_same(tmp_path):
    f = frame_from_arrays(echo_axis_arrays())
    write_frame(f, tmp_path / "flat.npz", layout="flat")
    g = read_frame(tmp_path / "flat.npz")
    for key in ("xyz", "echo", "ring", "intensity"):
        np.testing.assert_array_equal(g[key], f[key])


def test_points_array_layout():
    pts = np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.7]], np.float32)
    f = frame_from_arrays({"points": pts})
    np.testing.assert_array_equal(f.xyz, pts[:, :3])
    np.testing.assert_array_equal(f.intensity, pts[:, 3])


def test_missing_coordinates_list_the_keys():
    with pytest.raises(KeyError, match="keys in the file"):
        frame_from_arrays({"a": np.zeros(3)})


def test_synthetic_layouts_read_to_the_same_frame(tmp_path):
    cfg = dict(frames=1, n_beams=16, n_azimuth=256)
    synthetic.generate(tmp_path / "a", synthetic.Config(**cfg), log=None)
    synthetic.generate(tmp_path / "b", synthetic.Config(**cfg, layout="flat"), log=None)
    a = read_frame(tmp_path / "a" / "frame_000000.npz")
    b = read_frame(tmp_path / "b" / "frame_000000.npz")
    for key in ("xyz", "intensity", "echo", "ring"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a.meta["roll"] == b.meta["roll"]
