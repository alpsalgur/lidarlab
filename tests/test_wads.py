import json

import numpy as np
import pytest

from lidarlab.datasets import wads


def fake_sequence(root, n=3):
    seq = root / "sequences" / "11"
    (seq / "velodyne").mkdir(parents=True)
    (seq / "labels").mkdir()
    rng = np.random.default_rng(0)
    for k in range(n):
        pts = rng.normal(size=(100, 4)).astype(np.float32)
        pts.tofile(seq / "velodyne" / f"{k:06d}.bin")
        semantic = np.where(np.arange(100) < 10, 110, 40).astype(np.uint32)
        (semantic | (7 << 16)).tofile(seq / "labels" / f"{k:06d}.label")  # instance id 7
    np.savetxt(seq / "times.txt", [0.0, 0.1, 0.2])
    return seq


def test_convert_keeps_semantic_labels(tmp_path, capsys):
    fake_sequence(tmp_path)
    counts = wads.convert(tmp_path, tmp_path / "out")
    assert counts == {110: 30, 40: 270}
    with np.load(tmp_path / "out" / "frame_000002.npz") as f:
        assert f["label"].dtype == np.uint16
        assert np.all(f["label"][:10] == 110) and np.all(f["label"][10:] == 40)
        assert float(f["timestamp"]) == pytest.approx(0.2)
        assert np.all(f["echo_number"] == 1)
    info = json.loads((tmp_path / "out" / "recording.json").read_text())
    assert info["noise_labels"] == [110] and info["labelled_frames"] == 3
    assert "falling snow" in capsys.readouterr().out


def test_warns_when_the_noise_class_is_missing(tmp_path, capsys):
    fake_sequence(tmp_path)
    wads.convert(tmp_path, tmp_path / "out", noise_labels=(999,))
    assert "none of the noise labels" in capsys.readouterr().out


def test_several_sequences_need_a_choice(tmp_path):
    fake_sequence(tmp_path / "a")
    fake_sequence(tmp_path / "b")
    with pytest.raises(ValueError, match="several sequences"):
        wads.find_scans(tmp_path)
