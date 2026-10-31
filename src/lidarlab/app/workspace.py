"""Workspaces: the test run, frame, camera, display settings and pipeline in one file, to
come back to an analysis later or hand it to a colleague (``.lidarlab``, JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from .display import Display

VERSION = 1
SUFFIX = ".lidarlab"


def capture(w) -> dict:
    camera = w.view.renderer.GetActiveCamera()
    display = asdict(w.display)
    display["hidden_echoes"] = sorted(display["hidden_echoes"])
    try:
        pipeline = w.pipeline.to_dict()
    except ValueError:
        pipeline = None  # steps typed into the shell cannot be saved
    return {
        "lidarlab_workspace": VERSION,
        "run": str(w.rec.path) if w.rec is not None else None,
        "frame": w.index,
        "camera": {"position": list(camera.GetPosition()), "focal": list(camera.GetFocalPoint()),
                   "up": list(camera.GetViewUp()), "parallel": bool(camera.GetParallelProjection()),
                   "scale": camera.GetParallelScale(), "angle": camera.GetViewAngle()},
        "display": display,
        "pipeline": pipeline,
        "inspector_field": w.inspector.field.currentText(),
        "metric": w.metrics_strip.key,
    }


def save(w, path) -> Path:
    path = Path(path)
    if path.suffix != SUFFIX:
        path = path.with_name(path.name + SUFFIX)
    data = capture(w)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def restore(w, data: dict) -> None:
    """Open the workspace's test run and put everything back."""
    if not isinstance(data, dict) or "lidarlab_workspace" not in data:
        raise ValueError("not a LidarLab workspace")
    run = data.get("run")
    if run and (w.rec is None or Path(run).resolve() != w.rec.path.resolve()):
        if not Path(run).is_dir():
            raise FileNotFoundError(f"test run not found: {run}")
        if not w.open_run(run):
            raise RuntimeError(f"could not open {run}")
    if data.get("pipeline"):
        w.pipeline.load_dict(data["pipeline"], run_script=w.shell.run_file_now)
    saved = data.get("display", {})
    known = {f.name for f in fields(Display)}
    for key, value in saved.items():
        if key not in known:
            continue
        if key == "hidden_echoes":
            value = set(value)
        elif key == "clims":
            value = {k: tuple(v) for k, v in value.items()}
        setattr(w.display, key, value)
    w.apply_display()
    if data.get("inspector_field"):
        w.inspector.field.setCurrentText(data["inspector_field"])
    index = w.metrics_strip.metric_box.findData(data.get("metric"))
    if index >= 0:
        w.metrics_strip.metric_box.setCurrentIndex(index)
    w.show_frame(int(data.get("frame", 0)), wait=True)
    cam = data.get("camera")
    if cam:
        camera = w.view.renderer.GetActiveCamera()
        camera.SetParallelProjection(cam["parallel"])
        camera.SetPosition(*cam["position"])
        camera.SetFocalPoint(*cam["focal"])
        camera.SetViewUp(*cam["up"])
        camera.SetParallelScale(cam["scale"])
        camera.SetViewAngle(cam["angle"])
        w.view.render()


def load(w, path) -> None:
    restore(w, json.loads(Path(path).read_text(encoding="utf-8")))
