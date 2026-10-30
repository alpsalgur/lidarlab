import json

import numpy as np

from lidarlab.datasets import kitti


def fake_drive(raw_root):
    """Two-frame drive folder laid out like an extracted KITTI raw zip."""
    src = raw_root / "2011_09_26" / "2011_09_26_drive_9999_sync"
    (src / "velodyne_points" / "data").mkdir(parents=True)
    (src / "oxts" / "data").mkdir(parents=True)
    for k in range(2):
        pts = np.arange(40, dtype=np.float32).reshape(10, 4) + k
        pts.tofile(src / "velodyne_points" / "data" / f"{k:010d}.bin")
        oxts = np.zeros(30)
        oxts[3:6] = [0.01 * k, 0.02, -1.5]
        np.savetxt(src / "oxts" / "data" / f"{k:010d}.txt", oxts[None])
    (src / "velodyne_points" / "timestamps.txt").write_text(
        "2011-09-26 13:02:25.000000000\n2011-09-26 13:02:25.100000000\n"
    )
    return src


def test_convert(tmp_path):
    src = fake_drive(tmp_path / "raw")
    assert kitti.drive_dir(tmp_path / "raw", "2011_09_26_drive_9999") == src
    assert kitti.convert(src, tmp_path / "out", log=None) == 2
    with np.load(tmp_path / "out" / "frame_000001.npz") as f:
        np.testing.assert_array_equal(f["x"], np.arange(1, 41, 4, dtype=np.float32))
        np.testing.assert_array_equal(f["intensity"], np.arange(4, 44, 4, dtype=np.float32))
        assert f["echo_number"].dtype == np.uint8
        assert np.all(f["echo_number"] == 1)
        np.testing.assert_allclose([f["roll"], f["pitch"], f["yaw"]], [0.01, 0.02, -1.5],
                                   rtol=1e-6)
    with np.load(tmp_path / "out" / "frame_000000.npz") as f0, \
            np.load(tmp_path / "out" / "frame_000001.npz") as f1:
        assert abs(float(f1["timestamp"] - f0["timestamp"]) - 0.1) < 1e-6
    info = json.loads((tmp_path / "out" / "recording.json").read_text())
    assert info["dataset"] == "KITTI raw" and info["frames"] == 2


def test_parse_timestamp():
    assert kitti.parse_timestamp("1970-01-01 00:00:01.500000000") == 1.5
