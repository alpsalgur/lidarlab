"""Processing steps applied to every frame before it is shown.

A step is a function or a ``Filter`` that takes a Frame and returns one of:

- a Frame (e.g. ``f[f.echo != 0]`` or a frame with changed fields),
- a boolean keep-mask with one value per point,
- None, after assigning fields on the frame it was given.

    @lab.step
    def remove_cover_echo(f):
        return f.echo != 0

    pipe.add(lab.filters.DROR(beta=3))      # built-in filter with parameters
    pipe["dror"].set(min_neighbors=2)
    pipe.save("denoise.yaml")               # preset: steps, order and parameter values
"""

from __future__ import annotations

import inspect
import runpy
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .filter import REGISTRY, Filter, FunctionFilter, Param
from .frame import Frame

PRESET_VERSION = 1


@dataclass
class Stats:
    """What one step did to one frame."""

    n_in: int
    n_out: int | None = None
    ms: float | None = None
    error: Exception | None = None


@dataclass
class Processed:
    """A frame after the pipeline: the output, per-step stats and the pipeline version."""

    frame: Frame
    stats: dict[str, Stats] = field(default_factory=dict)
    version: int = 0

    @property
    def ms(self) -> float:
        return sum(s.ms or 0.0 for s in self.stats.values())


class Step:
    """One function or filter in a pipeline, with statistics from the frame shown last."""

    __slots__ = ("name", "func", "enabled", "ms", "n_in", "n_out", "error", "_owner")

    def __init__(self, name: str, func: Callable, enabled: bool = True, owner=None):
        self.name = name
        self.func = func
        self.enabled = enabled
        self.ms: float | None = None
        self.n_in: int | None = None
        self.n_out: int | None = None
        self.error: Exception | None = None
        self._owner = owner

    @property
    def params(self) -> dict[str, Param]:
        return self.func.params if isinstance(self.func, Filter) else {}

    @property
    def values(self) -> dict:
        return dict(self.func.values) if isinstance(self.func, Filter) else {}

    def set(self, **values) -> None:
        """Change parameter values: ``pipe["dror"].set(beta=4)``."""
        if not isinstance(self.func, Filter):
            raise TypeError(f"step {self.name!r} is a plain function without parameters; "
                            "use @lab.filter to give it some")
        self.func.set(**values)
        if self._owner is not None:
            self._owner.changed()

    def reset(self) -> None:
        """Parameters back to their defaults."""
        if isinstance(self.func, Filter):
            self.func.reset()
            if self._owner is not None:
                self._owner.changed()

    @property
    def description(self) -> str:
        if isinstance(self.func, Filter):
            return self.func.description
        return inspect.cleandoc(self.func.__doc__ or "")

    @property
    def source(self) -> tuple[str, int] | None:
        """(file, line) where the function is defined, if it comes from a file."""
        func = self.func.func if isinstance(self.func, FunctionFilter) else self.func
        if isinstance(func, Filter):
            func = type(func)
        try:
            path = inspect.getsourcefile(func)
            line = inspect.getsourcelines(func)[1]
        except (OSError, TypeError):
            return None
        return (path, line) if path else None

    def __repr__(self) -> str:
        state = "on" if self.enabled else "off"
        if self.values:
            state += ", " + ", ".join(f"{k}={v!r}" for k, v in self.values.items())
        if self.n_out is not None:
            state += f", {self.n_in:,} -> {self.n_out:,} pts, {self.ms:.1f} ms"
        return f"Step({self.name!r}, {state})"


class StepError(RuntimeError):
    """A step raised; the original exception is the ``__cause__``."""

    def __init__(self, step: str, error: Exception):
        super().__init__(f"step {step!r} failed: {type(error).__name__}: {error}")
        self.step = step


def _default_name(func) -> str:
    if isinstance(func, Filter):
        return func.name
    return getattr(func, "__name__", "step")


class Pipeline:
    """Ordered processing steps. Calling it runs the enabled steps on a frame."""

    def __init__(self):
        self.steps: list[Step] = []
        self.version = 0  # bumped on every change, including parameter values
        self._listeners: list[Callable[[], None]] = []

    def add(self, func, name: str | None = None, enabled: bool = True) -> Step:
        """Append ``func`` (a function, a Filter or a Filter class).

        A step with the same name is replaced in place. If both are filters, parameters
        whose default did not change keep their current value.
        """
        if isinstance(func, type) and issubclass(func, Filter):
            func = func()
        name = name or _default_name(func)
        for s in self.steps:
            if s.name == name:
                _keep_tuned_values(s.func, func)
                s.func, s.enabled, s.error = func, enabled, None
                break
        else:
            s = Step(name, func, enabled, owner=self)
            self.steps.append(s)
        self._changed()
        return s

    def unique_name(self, base: str) -> str:
        """``base``, or ``base_2``, ``base_3``… if a step already has that name."""
        name, k = base, 2
        while name in self:
            name, k = f"{base}_{k}", k + 1
        return name

    def remove(self, name: str) -> None:
        self.steps.remove(self[name])
        self._changed()

    def clear(self) -> None:
        self.steps.clear()
        self._changed()

    def enable(self, name: str, on: bool = True) -> None:
        self[name].enabled = on
        self._changed()

    def set(self, name: str, **values) -> None:
        """Change parameter values of step ``name``."""
        self[name].set(**values)

    def move(self, name: str, offset: int) -> None:
        """Move a step up (offset < 0) or down."""
        s = self[name]
        i = self.steps.index(s)
        j = min(max(i + offset, 0), len(self.steps) - 1)
        self.steps.insert(j, self.steps.pop(i))
        self._changed()

    def __getitem__(self, key) -> Step:
        if isinstance(key, int):
            return self.steps[key]
        for s in self.steps:
            if s.name == key:
                return s
        names = ", ".join(s.name for s in self.steps) or "none"
        raise KeyError(f"no step {key!r} (steps: {names})")

    def __contains__(self, name) -> bool:
        return any(s.name == name for s in self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __repr__(self) -> str:
        if not self.steps:
            return "Pipeline(no steps)"
        return "Pipeline(\n" + "".join(f"  {k}. {s!r}\n" for k, s in enumerate(self.steps, 1)) + ")"

    @property
    def active(self) -> bool:
        return any(s.enabled for s in self.steps)

    def snapshot(self) -> Pipeline:
        """An independent copy: later changes here (sliders, scripts run again) do not reach
        it. Filters are copied with their values; plain functions are shared."""
        copy = Pipeline()
        for s in self.steps:
            func = deepcopy(s.func) if isinstance(s.func, Filter) else s.func
            copy.steps.append(Step(s.name, func, s.enabled, owner=copy))
        return copy

    # running

    def process(self, frame: Frame, errors: str = "skip") -> Processed:
        """Run the enabled steps and return the output with per-step stats, without
        touching the steps (safe from a background thread). With ``errors="skip"`` a
        failing step passes its input on; with "raise" it raises StepError."""
        version = self.version
        frame = frame.view()  # steps may assign fields; keep the caller's frame unchanged
        stats: dict[str, Stats] = {}
        for s in [s for s in self.steps if s.enabled]:
            st = stats[s.name] = Stats(len(frame))
            start = time.perf_counter()
            try:
                frame = _as_frame(s.func(frame), frame)
            except Exception as e:
                st.error = e
                if errors == "raise":
                    raise StepError(s.name, e) from e
                continue
            st.ms = (time.perf_counter() - start) * 1e3
            st.n_out = len(frame)
        return Processed(frame, stats, version)

    def record(self, stats: dict[str, Stats]) -> None:
        """Show ``stats`` (of the frame on screen) on the steps."""
        for s in self.steps:
            st = stats.get(s.name)
            if st is not None:
                s.n_in, s.n_out, s.ms, s.error = st.n_in, st.n_out, st.ms, st.error

    def __call__(self, frame: Frame, errors: str = "raise") -> Frame:
        """Run the enabled steps and keep their stats on the steps."""
        try:
            result = self.process(frame, errors="skip" if errors == "skip" else "raise")
        except StepError as e:
            self[e.step].error = e.__cause__
            raise
        self.record(result.stats)
        return result.frame

    # listeners

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` after every change to the steps."""
        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        self._listeners.remove(callback)

    def changed(self) -> None:
        """Tell listeners that something a step depends on changed (e.g. a parameter)."""
        self._changed()

    def _changed(self) -> None:
        self.version += 1
        for callback in list(self._listeners):
            callback()

    # presets

    def to_dict(self) -> dict:
        """Preset: built-in filters by key, script steps by the file that defines them."""
        entries = []
        for s in self.steps:
            entry: dict = {"name": s.name, "enabled": s.enabled}
            if isinstance(s.func, Filter) and type(s.func).key in REGISTRY:
                entry["filter"] = type(s.func).key
            else:
                source = s.source
                if source is None or source[0].startswith("<"):
                    raise ValueError(f"step {s.name!r} was defined in the shell; save it in a "
                                     "script file to keep it in a preset")
                entry["script"] = str(Path(source[0]).resolve())
            if s.values:
                entry["params"] = {k: _plain(v) for k, v in s.values.items()}
            entries.append(entry)
        return {"lidarlab_pipeline": PRESET_VERSION, "steps": entries}

    def save(self, path) -> Path:
        path = Path(path)
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        path.write_text("# LidarLab pipeline preset\n" + text, encoding="utf-8")
        return path

    def load_dict(self, data: dict, run_script: Callable[[str], None] | None = None) -> None:
        """Replace the steps with a preset. Script steps are defined by running their file
        once (``run_script(path)``, default ``runpy.run_path``); its decorators add them."""
        if not isinstance(data, dict) or "steps" not in data:
            raise ValueError("not a LidarLab pipeline preset (no 'steps')")
        entries = data["steps"] or []
        listeners, self._listeners = self._listeners, []  # one notification at the end
        try:
            self.steps.clear()
            scripts: list[str] = []
            for e in entries:
                if "filter" in e:
                    cls = REGISTRY.get(e["filter"])
                    if cls is None:
                        raise ValueError(f"unknown filter {e['filter']!r}")
                    self.add(cls(), name=e.get("name"))
                elif "script" in e and e["script"] not in scripts:
                    scripts.append(e["script"])
            _targets.append(self)  # the scripts' @lab.step / @lab.filter add steps here
            try:
                for path in scripts:
                    if not Path(path).is_file():
                        raise FileNotFoundError(f"script of a preset step not found: {path}")
                    (run_script or runpy.run_path)(path)
            finally:
                _targets.pop()
            order = []
            for e in entries:
                name = e.get("name") or e.get("filter")
                if name not in self:
                    raise ValueError(f"step {name!r} was not defined by {e.get('script')}")
                s = self[name]
                s.enabled = bool(e.get("enabled", True))
                if e.get("params"):
                    s.func.set(**e["params"])
                order.append(s)
            self.steps[:] = order
        finally:
            self._listeners = listeners
        self._changed()

    def load(self, path, run_script: Callable[[str], None] | None = None) -> None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.load_dict(data, run_script)


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value


def _keep_tuned_values(old, new) -> None:
    if not (isinstance(old, Filter) and isinstance(new, Filter)):
        return
    for k, p in new.params.items():
        q = old.params.get(k)
        if q is not None and q.default == p.default and q.kind == p.kind:
            try:
                new.values[k] = p.check(old.values[k])
            except (TypeError, ValueError):
                pass


def _as_frame(result, frame: Frame) -> Frame:
    if result is None:
        return frame
    if isinstance(result, Frame):
        return result
    mask = np.asarray(result)
    if mask.dtype == bool and mask.shape == (len(frame),):
        return frame[mask]
    raise TypeError(f"a step must return a Frame, a boolean mask of {len(frame)} values or None, "
                    f"got {type(result).__name__} with shape {mask.shape}")


pipeline = Pipeline()  # the pipeline the viewer applies; ``lab.pipeline`` / ``pipe``
_targets: list[Pipeline] = []  # pipelines loading a preset, innermost last


def default_pipeline() -> Pipeline:
    """Where ``@lab.step`` and ``@lab.filter`` add steps: ``lab.pipeline``, or the
    pipeline whose preset is running the script."""
    return _targets[-1] if _targets else pipeline


def step(func: Callable | None = None, *, name: str | None = None, enabled: bool = True,
         to: Pipeline | None = None):
    """Decorator: add ``func`` as a step of ``to`` (default: ``lab.pipeline``).

    Running the defining script again replaces the step instead of adding a second copy.
    """

    def register(f: Callable) -> Callable:
        (default_pipeline() if to is None else to).add(f, name=name, enabled=enabled)
        return f

    return register if func is None else register(func)
