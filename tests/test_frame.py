import pickle

import numpy as np
import pytest

from lidarlab import Frame


def make() -> Frame:
    xyz = np.arange(15, dtype=np.float32).reshape(5, 3)
    fields = {
        "intensity": np.arange(5, dtype=np.float32) * 10,
        "echo": np.array([0, 1, 2, 1, 1], np.uint8),
    }
    return Frame(xyz, fields, meta={"frame": 7}, index=[0, 3, 4, 1, 5])


def test_fields_read_as_attributes_and_keys():
    f = make()
    assert f.intensity is f["intensity"]
    np.testing.assert_array_equal(f.x, [0, 3, 6, 9, 12])
    np.testing.assert_array_equal(f["z"], [2, 5, 8, 11, 14])
    assert {"intensity", "echo"} <= set(dir(f))
    assert "echo" in f and "xyz" in f and "nope" not in f
    with pytest.raises(AttributeError, match="no field 'nope'"):
        _ = f.nope
    assert repr(f) == "Frame #7(5 points; fields: intensity, echo)"


def test_mask_subset_keeps_fields_index_and_meta():
    f = make()
    g = f[f.echo != 0]
    assert len(g) == 4
    np.testing.assert_array_equal(g.index, [3, 4, 1, 5])
    np.testing.assert_array_equal(g.intensity, [10, 20, 30, 40])
    assert g.meta == f.meta and g.meta is not f.meta
    with pytest.raises(IndexError):
        f[np.ones(3, bool)]


def test_integer_index_returns_one_point():
    assert make()[1] == {"x": 3.0, "y": 4.0, "z": 5.0, "intensity": 10.0, "echo": 1}


def test_assigning_fields_leaves_shared_arrays_alone():
    f = make()
    f.intensity.flags.writeable = False
    g = f.view()
    g["intensity"] = g.intensity * 2
    g.ring = 3  # attribute assignment adds a field; scalars are broadcast
    np.testing.assert_array_equal(f.intensity, [0, 10, 20, 30, 40])
    np.testing.assert_array_equal(g.intensity, [0, 20, 40, 60, 80])
    assert g.ring.tolist() == [3] * 5 and "ring" not in f.fields
    with pytest.raises(ValueError, match="needs 5 values"):
        g["bad"] = np.zeros(4)
    with pytest.raises(ValueError, match="read-only"):
        f.intensity[0] = 1
    c = f.copy()
    c.intensity[0] = 1
    assert f.intensity[0] == 0


def test_coordinate_assignment_copies_xyz():
    f = make()
    xyz = f.xyz
    f.z = 0
    assert np.all(f.z == 0) and xyz[0, 2] == 2


def test_derived_values():
    f = Frame([[3, 4, 0], [0, 2, 2]], index=[1, 4], meta={"shape": (3, 2)})
    np.testing.assert_allclose(f.range, [5, np.sqrt(8)], rtol=1e-6)
    np.testing.assert_allclose(f.azimuth, [np.degrees(np.arctan2(4, 3)), 90], rtol=1e-6)
    np.testing.assert_allclose(f.elevation, [0, 45], rtol=1e-6)
    np.testing.assert_array_equal(f.pulse, [1, 0])
    with pytest.raises(ValueError, match="echo-axis"):
        _ = Frame(np.zeros((2, 3))).pulse


def test_pickle_roundtrip():
    f = make()
    g = pickle.loads(pickle.dumps(f))
    np.testing.assert_array_equal(g.xyz, f.xyz)
    np.testing.assert_array_equal(g.index, f.index)
    assert list(g.fields) == list(f.fields) and g.meta == f.meta
