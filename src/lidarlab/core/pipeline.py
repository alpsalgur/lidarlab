"""Processing steps applied to every frame before it is shown.

A step is a function that takes a Frame and returns one of:

- a Frame (e.g. ``f[f.echo != 0]`` or a frame with changed fields),
- a boolean keep-mask with one value per point,
- None, after assigning fields on the frame it was given.

    @lab.step
    def remove_cover_echo(f):
        return f.echo != 0
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable

import numpy as np

from .frame import Frame


class Step:
    """One function in a pipeline, with statistics from its last run."""

    __slots__ = ("name", "func", "enabled", "ms", "n_in", "n_out", "error")

    def __init__(self, name: str, func: Callable, enabled: bool = True):
        self.name = name
        self.func = func
        self.enabled = enabled
        self.ms: float | None = None
        self.n_in: int | None = None
        self.n_out: int | None = None
        self.error: Exception | None = None

    @property
    def source(self) -> tuple[str, int] | None:
        """(file, line) where the function is defined, if it comes from a file."""
        try:
            path = inspect.getsourcefile(self.func)
            line = inspect.getsourcelines(self.func)[1]
        except (OSError, TypeError):
            return None
        return (path, line) if path else None

    def __repr__(self) -> str:
        state = "on" if self.enabled else "off"
        if self.n_out is not None:
            state += f", {self.n_in:,} -> {self.n_out:,} pts, {self.ms:.1f} ms"
        return f"Step({self.name!r}, {state})"


class StepError(RuntimeError):
    """A step raised; the original exception is the ``__cause__``."""

    def __init__(self, step: str, error: Exception):
        super().__init__(f"step {step!r} failed: {type(error).__name__}: {error}")
        self.step = step


class Pipeline:
    """Ordered processing steps. Calling it runs the enabled steps on a frame."""

    def __init__(self):
        self.steps: list[Step] = []
        self.version = 0  # bumped on every change
        self._listeners: list[Callable[[], None]] = []

    def add(self, func: Callable, name: str | None = None, enabled: bool = True) -> Step:
        """Append ``func``; a step with the same name is replaced in place."""
        name = name or getattr(func, "__name__", "step")
        for s in self.steps:
            if s.name == name:
                s.func, s.enabled, s.error = func, enabled, None
                break
        else:
            s = Step(name, func, enabled)
            self.steps.append(s)
        self._changed()
        return s

    def remove(self, name: str) -> None:
        self.steps.remove(self[name])
        self._changed()

    def clear(self) -> None:
        self.steps.clear()
        self._changed()

    def enable(self, name: str, on: bool = True) -> None:
        self[name].enabled = on
        self._changed()

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

    def __call__(self, frame: Frame, errors: str = "raise") -> Frame:
        """Run the enabled steps. With ``errors="skip"`` a failing step passes its input on
        and keeps the exception in ``step.error``."""
        frame = frame.view()  # steps may assign fields; keep the caller's frame unchanged
        for s in self.steps:
            if not s.enabled:
                continue
            start = time.perf_counter()
            s.n_in = len(frame)
            try:
                frame = _as_frame(s.func(frame), frame)
            except Exception as e:
                s.error, s.ms, s.n_out = e, None, None
                if errors == "raise":
                    raise StepError(s.name, e) from e
                continue
            s.error = None
            s.ms = (time.perf_counter() - start) * 1e3
            s.n_out = len(frame)
        return frame

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


def step(func: Callable | None = None, *, name: str | None = None, enabled: bool = True,
         to: Pipeline | None = None):
    """Decorator: add ``func`` as a step of ``to`` (default: ``lab.pipeline``).

    Running the defining script again replaces the step instead of adding a second copy.
    """

    def register(f: Callable) -> Callable:
        (pipeline if to is None else to).add(f, name=name, enabled=enabled)
        return f

    return register if func is None else register(func)
