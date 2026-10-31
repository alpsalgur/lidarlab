import json

import numpy as np

from lidarlab import Frame
from lidarlab.core import filters as F
from lidarlab.datasets import boreas

ROLL, PITCH, HEADING = 0.04, -0.03, 1.2  # lidar pose as Boreas stores it, rad
MOUNT = np.radians(-47.4)  # lidar turned about z relative to the vehicle, as in Boreas


def fake_sequence(root, n_frames=2):
    """Level ground 2 m below the sensor, seen by a tilted lidar mounted at -47.4°."""
    src = root / "boreas-2099-01-01-00-00"
    (src / "applanix").mkdir(parents=True)
    (src / "calib").mkdir()
    (src / "lidar").mkdir()
    c, s = np.cos(MOUNT), np.sin(MOUNT)
    t_applanix_lidar = np.eye(4)
    # vehicle <- lidar = VEHICLE_FROM_APPLANIX @ applanix <- lidar should be Rz(MOUNT)
    t_applanix_lidar[:3, :3] = boreas.VEHICLE_FROM_APPLANIX.T @ np.array(
        [[c, -s, 0], [s, c, 0], [0, 0, 1]])
    np.savetxt(src / "calib" / "T_applanix_lidar.txt", t_applanix_lidar)
    rows = []
    rng = np.random.default_rng(0)
    for k in range(n_frames):
        stamp = 1_611_678_000_000_000 + k * 100_000
        enu = np.c_[rng.uniform(-20, 20, (500, 2)), np.full(500, -2.0)]
        c_enu_lidar = boreas.enu_from_lidar(ROLL, PITCH, HEADING)
        lidar = enu @ c_enu_lidar  # rows of C^T p
        pts = np.c_[lidar, np.full(500, 40.0), np.arange(500) % 128,
                    np.linspace(-0.05, 0.05, 500)].astype(np.float32)
        pts.tofile(src / "lidar" / f"{stamp}.bin")
        rows.append([stamp, 0, 0, 0, 0, 0, 0, ROLL, PITCH, HEADING, 0, 0, 0])
    header = "GPSTime,easting,northing,altitude,vel_east,vel_north,vel_up,roll,pitch,heading," \
             "angvel_z,angvel_y,angvel_x"
    np.savetxt(src / "applanix" / "lidar_poses.csv", np.array(rows), delimiter=",",
               header=header, comments="", fmt=["%d"] + ["%.10g"] * 12)  # 16-digit stamps
    return src


def test_convert_levels_and_turns_to_the_vehicle(tmp_path):
    src = fake_sequence(tmp_path)
    assert boreas.convert(src, tmp_path / "out", log=None) == 2
    with np.load(tmp_path / "out" / "frame_000001.npz") as f:
        frame = Frame(np.c_[f["x"], f["y"], f["z"]], {"intensity": f["intensity"]},
                      {"roll": f["roll"], "pitch": f["pitch"]})
        assert f["ring"].dtype == np.uint8 and f["ring"].max() == 127
        assert f["echo_number"].dtype == np.uint8 and np.all(f["echo_number"] == 1)
        assert f["timestamp"] == 1_611_678_000.1
        assert abs(float(f["t"].min()) + 0.05) < 1e-6
    # the stored roll/pitch level the tilted scan: the ground is flat at -2 m again
    level = F.Level()(frame)
    np.testing.assert_allclose(level.z, -2.0, atol=1e-4)
    info = json.loads((tmp_path / "out" / "recording.json").read_text())
    assert info["dataset"] == "Boreas" and "CC BY 4.0" in info["license"]


def test_attitude_of_a_level_vehicle_heading_north():
    c = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # x north, y west
    roll, pitch, yaw = boreas.attitude(c)
    assert abs(roll) < 1e-12 and abs(pitch) < 1e-12
    assert abs(yaw - np.pi / 2) < 1e-12
