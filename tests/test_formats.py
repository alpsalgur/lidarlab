import numpy as np

from lidarlab import Frame, write_pcd, write_ply

PLY_DTYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "char": "i1", "ushort": "<u2",
              "short": "<i2", "uint": "<u4", "int": "<i4"}
PCD_DTYPES = {("F", "4"): "<f4", ("F", "8"): "<f8", ("U", "1"): "u1", ("U", "2"): "<u2",
              ("U", "4"): "<u4", ("I", "1"): "i1", ("I", "2"): "<i2", ("I", "4"): "<i4"}


def frame():
    rng = np.random.default_rng(0)
    return Frame(rng.normal(size=(50, 3)), {
        "intensity": rng.uniform(0, 255, 50).astype(np.float32),
        "echo": rng.integers(0, 3, 50).astype(np.uint8),
        "label": rng.integers(0, 120, 50).astype(np.uint16),
        "count": np.arange(50, dtype=np.int64),
        "flag": np.arange(50) % 2 == 0,
    })


def split_header(data: bytes, end: bytes) -> tuple[list[str], bytes]:
    head, _, body = data.partition(end)
    return head.decode("ascii").splitlines(), body


def test_ply_round_trip(tmp_path):
    f = frame()
    path = write_ply(f, tmp_path / "f.ply")
    lines, body = split_header(path.read_bytes(), b"end_header\n")
    assert lines[:2] == ["ply", "format binary_little_endian 1.0"]
    assert "element vertex 50" in lines
    props = [line.split()[1:] for line in lines if line.startswith("property")]
    assert [name for _, name in props] == ["x", "y", "z", "intensity", "echo", "label", "count",
                                           "flag"]
    table = np.frombuffer(body, np.dtype([(n, PLY_DTYPES[t]) for t, n in props]))
    np.testing.assert_array_equal(np.c_[table["x"], table["y"], table["z"]], f.xyz)
    np.testing.assert_array_equal(table["label"], f.label)
    assert table.dtype["count"] == np.dtype("<i4") and table["count"][-1] == 49
    assert table["flag"].tolist()[:3] == [1, 0, 1]


def test_pcd_round_trip(tmp_path):
    f = frame()
    path = write_pcd(f, tmp_path / "f.pcd")
    lines, body = split_header(path.read_bytes(), b"DATA binary\n")
    header = dict(line.split(" ", 1) for line in lines if not line.startswith("#"))
    names, sizes, types = (header[k].split() for k in ("FIELDS", "SIZE", "TYPE"))
    assert header["POINTS"] == "50" and header["WIDTH"] == "50" and header["HEIGHT"] == "1"
    table = np.frombuffer(body, np.dtype([(n, PCD_DTYPES[(t, s)])
                                          for n, s, t in zip(names, sizes, types, strict=True)]))
    np.testing.assert_array_equal(table["intensity"], f.intensity)
    np.testing.assert_array_equal(table["echo"], f.echo)
    assert len(body) == 50 * table.dtype.itemsize
