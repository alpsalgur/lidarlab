import json

import numpy as np
import pytest

import lidarlab as lab
from lidarlab import Pipeline
from lidarlab.core import batch
from lidarlab.core import filters as F
from lidarlab.datasets import synthetic


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    path = tmp_path_factory.mktemp("run")
    synthetic.generate(path, synthetic.Config(frames=4, n_beams=32, n_azimuth=512), log=None)
    return path


def rain_pipeline() -> Pipeline:
    p = Pipeline()
    p.add(F.EchoSelect())
    p.add(F.DROR(beta=4.0))
    return p


def test_export_writes_a_test_run_in_the_source_layout(run, tmp_path):
    out = tmp_path / "out"
    report = lab.export_run(run, out, rain_pipeline(), workers=2)
    assert report.written == 4 and not report.failed and not report.cancelled
    assert sorted(p.name for p in out.glob("*.npz")) == sorted(p.name for p in run.glob("*.npz"))
    with np.load(run / "frame_000002.npz") as src, np.load(out / "frame_000002.npz") as dst:
        assert dst["x"].shape == src["x"].shape  # echo-axis kept: removed points are NaN
        assert np.all(np.isnan(dst["x"][0]))  # the cover echo is gone
        kept = np.isfinite(dst["x"])
        np.testing.assert_array_equal(dst["intensity"][kept], src["intensity"][kept])
        assert dst["roll"] == src["roll"]
    expected = rain_pipeline()(lab.read_frame(run / "frame_000002.npz"))
    assert np.count_nonzero(np.isfinite(np.load(out / "frame_000002.npz")["x"])) == len(expected)
    info = json.loads((out / "recording.json").read_text())
    assert info["processed"]["steps"] == ["echo", "dror"] and info["processed"]["written"] == 4
    assert info["labels"]["5"] == "rain"  # the source's description is kept
    assert (out / "pipeline.yaml").exists()
    log = (out / "export.log").read_text().splitlines()
    assert log[0].startswith("frame\tfile") and len(log) == 5
    assert lab.open(out)[2].fields.keys() == lab.open(run)[2].fields.keys()


def test_threads_and_processes_give_the_same_frames(run, tmp_path):
    lab.export_run(run, tmp_path / "threads", rain_pipeline(), workers=2)
    lab.export_run(run, tmp_path / "processes", rain_pipeline(), workers=2, processes=True)
    for path in (tmp_path / "threads").glob("*.npz"):
        with np.load(path) as a, np.load(tmp_path / "processes" / path.name) as b:
            for key in a.files:
                assert np.array_equal(a[key], b[key], equal_nan=True), (path.name, key)


def test_range_cancel_and_formats(run, tmp_path):
    report = lab.export_run(run, tmp_path / "range", frames="1:3")
    assert [r.index for r in report.results] == [1, 2]
    assert sorted(p.name for p in (tmp_path / "range").glob("*.npz")) == [
        "frame_000001.npz", "frame_000002.npz"]
    report = lab.export_run(run, tmp_path / "cancel", workers=1, progress=lambda *a: False)
    assert report.cancelled and report.written < 4
    report = lab.export_run(run, tmp_path / "ply", rain_pipeline(), fmt="ply", frames=[0])
    assert (tmp_path / "ply" / "frame_000000.ply").read_bytes().startswith(b"ply\n")
    assert batch.select_frames("-2:", 4) == [2, 3] and batch.select_frames([-1], 4) == [3]


def test_failing_frames_are_not_written(run, tmp_path):
    p = Pipeline()

    def breaks_on_frame_2(f):
        if f.meta["frame"] == 2:
            raise RuntimeError("bad frame")

    p.add(breaks_on_frame_2)
    report = lab.export_run(run, tmp_path / "out", p)
    assert [r.index for r in report.failed] == [2]
    assert "RuntimeError: bad frame" in report.failed[0].error
    assert not (tmp_path / "out" / "frame_000002.npz").exists()
    assert report.written == 3
    assert "bad frame" in (tmp_path / "out" / "export.log").read_text()


def test_output_folder_rules(run, tmp_path):
    with pytest.raises(ValueError, match="test run itself"):
        lab.export_run(run, run)
    out = tmp_path / "out"
    lab.export_run(run, out)
    with pytest.raises(FileExistsError):
        lab.export_run(run, out)
    (out / "notes.txt").write_text("mine")
    lab.export_run(run, out, frames="0:1", overwrite=True)
    assert [p.name for p in out.glob("*.npz")] == ["frame_000000.npz"]  # no stale frames
    assert (out / "notes.txt").exists()  # other files are left alone


def test_steps_from_the_shell_need_threads(run, tmp_path):
    p = Pipeline()
    exec("def from_shell(f):\n    return None", ns := {})
    p.add(ns["from_shell"])
    with pytest.raises(ValueError, match="processes need"):
        lab.export_run(run, tmp_path / "a", p, processes=True)
    assert lab.export_run(run, tmp_path / "b", p).written == 4
    assert not (tmp_path / "b" / "pipeline.yaml").exists()


def test_command_line(run, tmp_path, capsys):
    preset = tmp_path / "preset.yaml"
    rain_pipeline().save(preset)
    code = batch.main([str(run), str(tmp_path / "out"), "-p", str(preset), "--frames", "0:2"])
    assert code == 0
    assert "2/2 frames written" in capsys.readouterr().out
    assert batch.main([str(run), str(tmp_path / "out")]) == 2  # not empty
    assert "not empty" in capsys.readouterr().err
