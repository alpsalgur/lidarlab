import numpy as np
import pytest

import lidarlab as lab
from lidarlab import Frame, Param, Pipeline
from lidarlab.core import filters as F
from lidarlab.core.filter import REGISTRY, FunctionFilter


def grid_plane(n=60, spacing=0.1, z=-1.8, x0=3.0):
    """Dense ground patch: n × n points."""
    u, v = np.meshgrid(np.arange(n) * spacing + x0, np.arange(n) * spacing - n * spacing / 2)
    return np.c_[u.ravel(), v.ravel(), np.full(u.size, z)]


def scene(rng=None):
    """Ground patch plus 20 isolated points floating above it."""
    rng = rng or np.random.default_rng(0)
    ground = grid_plane()
    drops = np.c_[rng.uniform(4, 8, 20), rng.uniform(-2, 2, 20), rng.uniform(0, 1.5, 20)]
    xyz = np.vstack([ground, drops])
    intensity = np.r_[np.full(len(ground), 50.0), np.full(20, 5.0)].astype(np.float32)
    return Frame(xyz, {"intensity": intensity}, {"frame": 7}), len(ground)


def echo_frame():
    """Echo-axis style frame: 4 pulses × 3 echoes, some missing."""
    # pulse:     0  0  0  1  1  2  2  2  3
    echo = np.array([0, 1, 2, 1, 2, 0, 1, 2, 1], np.uint8)
    pulse = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3])
    intensity = np.array([90, 5, 60, 40, 30, 80, 70, 10, 50], np.float32)
    index = echo * 4 + pulse
    xyz = np.c_[np.arange(9.0), np.zeros(9), np.zeros(9)]
    return Frame(xyz, {"echo": echo, "intensity": intensity}, {"shape": (3, 4)}, index)


def kept(flt, f):
    keep = flt(f)
    return np.flatnonzero(keep).tolist()


def test_echo_modes():
    f = echo_frame()
    assert kept(F.EchoSelect(), f) == [1, 2, 3, 4, 6, 7, 8]
    assert kept(F.EchoSelect(mode="first real echo"), f) == [1, 3, 6, 8]
    assert kept(F.EchoSelect(mode="last real echo"), f) == [2, 4, 7, 8]
    assert kept(F.EchoSelect(mode="strongest real echo"), f) == [2, 3, 6, 8]
    assert kept(F.EchoSelect(mode="only echo n", n=2), f) == [2, 4, 7]


def test_echo_modes_without_pulse_ids():
    f = Frame(np.zeros((3, 3)), {"echo": np.array([0, 1, 1], np.uint8)})
    assert kept(F.EchoSelect(mode="last real echo"), f) == [1, 2]  # one real echo per pulse
    f = Frame(np.zeros((3, 3)), {"echo": np.array([0, 1, 2], np.uint8)})
    with pytest.raises(ValueError, match="pulse ids"):
        F.EchoSelect(mode="last real echo")(f)


def test_weak_first_echo():
    f = echo_frame()
    # pulse 0: first real echo 1 (5) is weak and echo 2 follows -> removed
    # pulse 2: first real echo 1 (70) is strong -> kept; pulse 3 has one echo -> kept
    assert kept(F.WeakFirstEcho(max_intensity=20), f) == [0, 2, 3, 4, 5, 6, 7, 8]
    single = Frame(np.zeros((2, 3)), {"echo": np.ones(2, np.uint8)})
    assert F.WeakFirstEcho()(single) is None


def test_threshold_and_crop():
    f, _ = scene()
    near = F.Threshold(field="x", min=0, max=5)(f)
    assert np.all(f.x[near] <= 5) and np.all(f.x[~near] > 5)
    assert np.array_equal(F.Threshold(field="x", min=0, max=5, mode="remove inside")(f), ~near)
    box = F.CropBox(x_min=4, x_max=8, y_min=-2, y_max=2, z_min=-0.5, z_max=2)(f)
    assert box.sum() == 20  # exactly the floating points


@pytest.mark.parametrize("flt", [F.ROR(radius=0.25), F.DROR(alpha_deg=1.0), F.SOR(std_ratio=1.0),
                                 F.LIOR(max_intensity=10, radius=0.25)])
def test_outlier_filters_remove_isolated_points(flt):
    f, n_ground = scene()
    keep = flt(f)
    assert keep[:n_ground].mean() > 0.97
    assert not keep[n_ground:].any()


def test_dsor_threshold_grows_with_range():
    # the same sparse line (0.5 m spacing) at 5 m and at 50 m above dense ground
    line = np.c_[np.zeros(10), np.arange(10) * 0.5 - 2.25, np.zeros(10)]
    xyz = np.vstack([grid_plane(), line + [5, 0, 0], line + [50, 0, 0]])
    keep = F.DSOR(range_mult=0.5)(Frame(xyz))
    assert keep[:3600].mean() > 0.97
    assert not keep[3600:3610].any()  # near: sparse for its range
    assert keep[3610:].all()  # far: as sparse as expected there


def test_voxel_keeps_one_point_per_voxel():
    f, _ = scene()
    keep = F.VoxelGrid(size=0.2)(f)
    cells = np.floor(f.xyz[keep] / 0.2).astype(int)
    assert len(np.unique(cells, axis=0)) == keep.sum()
    assert keep.sum() < len(f) / 3


def test_level_undoes_roll_and_pitch():
    f, n_ground = scene()
    r = F.rotation(0.05, -0.03)
    tilted = Frame(f.xyz @ np.linalg.inv(r).T, f.fields, {"roll": 0.05, "pitch": -0.03})
    out = F.Level()(tilted)
    np.testing.assert_allclose(out.xyz, f.xyz, atol=1e-4)
    with pytest.raises(ValueError, match="roll"):
        F.Level()(Frame(f.xyz))


def test_ground_plane_found_under_tilt():
    f, n_ground = scene()
    keep = F.GroundPlane(max_range=50)(f)
    assert not keep[:n_ground].any() and keep[n_ground:].all()
    n, d = F.GroundPlane(max_range=50).plane(f)
    np.testing.assert_allclose(n, [0, 0, 1], atol=1e-6)
    assert d == pytest.approx(1.8, abs=1e-4)


def test_params_are_checked_and_clipped():
    dror = F.DROR(beta=100)
    assert dror.beta == 10.0  # clipped to the range
    dror.min_neighbors = 4.6
    assert dror.min_neighbors == 5 and isinstance(dror.min_neighbors, int)
    with pytest.raises(TypeError, match="no parameter 'radius'"):
        dror.set(radius=1)
    with pytest.raises(ValueError, match="not one of"):
        F.EchoSelect(mode="loudest")
    assert Param(True).check("off") is False


def test_every_builtin_filter_runs_on_empty_and_real_frames():
    f, _ = scene()
    empty = f[np.zeros(len(f), bool)]
    for key, cls in REGISTRY.items():
        if not issubclass(cls, FunctionFilter) and cls.__module__ == F.__name__:
            flt = cls()
            assert flt.title and flt.description, key
            if key == "level":
                continue  # needs roll/pitch
            for frame in (f, empty):
                result = Pipeline()
                result.add(flt)
                assert len(result(frame)) <= len(frame), key


def test_function_filter_decorator():
    p = Pipeline()

    @lab.filter(radius=(0.05, 2.0), mode=["keep", "drop"], to=p)
    def near(f, radius=0.5, mode="keep", flag=True):
        """Points near the sensor."""
        inside = f.range < radius * 10
        return inside if mode == "keep" else ~inside

    s = p["near"]
    assert set(s.params) == {"radius", "mode", "flag"}
    assert s.params["radius"].lo == 0.05 and s.params["mode"].choices == ("keep", "drop")
    assert s.params["flag"].kind == "bool"
    assert s.description == "Points near the sensor."
    f, _ = scene()
    assert len(p(f)) == np.count_nonzero(f.range < 5)
    s.set(mode="drop")
    assert len(p(f)) == np.count_nonzero(f.range >= 5)
    assert near(f, radius=0.1).dtype == bool  # still a plain function


def test_rerun_keeps_tuned_values_unless_the_default_changed():
    p = Pipeline()

    def define(default_radius):
        @lab.filter(radius=(0.0, 5.0), k=(1, 9), to=p)
        def f1(f, radius=default_radius, k=3):
            return None

    define(1.0)
    p.set("f1", radius=2.5, k=7)
    define(1.0)  # script run again, same defaults
    assert p["f1"].values == {"radius": 2.5, "k": 7}
    define(1.5)  # the default of radius changed in the script
    assert p["f1"].values == {"radius": 1.5, "k": 7}


def test_decorator_rejects_bad_specs():
    with pytest.raises(TypeError, match="needs a default"):
        @lab.filter(to=Pipeline())
        def a(f, radius):
            return None
    with pytest.raises(TypeError, match="no parameter"):
        @lab.filter(size=(0, 1), to=Pipeline())
        def b(f, radius=1.0):
            return None
