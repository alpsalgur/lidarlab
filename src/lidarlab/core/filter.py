"""Filters: pipeline steps with named parameters.

The app turns the parameters into sliders and boxes, presets save their values. Built-in
filters subclass ``Filter``; scripts write a function and decorate it:

    @lab.filter(radius=(0.05, 2.0), min_neighbors=(0, 20))
    def ror(f, radius=0.4, min_neighbors=2):
        ...
        return keep

Defaults come from the function signature, ranges from the decorator: ``(lo, hi)`` or
``(lo, hi, step)`` for numbers, a list of strings for a choice, or a ``Param``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from .frame import Frame

KINDS = ("float", "int", "bool", "choice", "field")
REGISTRY: dict[str, type[Filter]] = {}  # built-in filters by key


@dataclass(frozen=True)
class Param:
    """One filter parameter: its default, valid range or choices, and how to show it.

    ``kind`` "field" is a per-point field of the frame (its choices come from the data).
    """

    default: Any
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    kind: str = ""
    unit: str = ""
    help: str = ""

    def __post_init__(self):
        kind = self.kind or _infer_kind(self.default, self.choices)
        if kind not in KINDS:
            raise ValueError(f"unknown parameter kind {kind!r}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "choices", tuple(self.choices))

    def check(self, value):
        """``value`` converted to this parameter's type and clipped to its range."""
        if self.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if self.kind in ("choice", "field"):
            value = str(value)
            if self.kind == "choice" and value not in self.choices:
                raise ValueError(f"{value!r} is not one of {', '.join(self.choices)}")
            return value
        value = int(round(float(value))) if self.kind == "int" else float(value)
        if self.lo is not None and value < self.lo:
            value = type(value)(self.lo)
        if self.hi is not None and value > self.hi:
            value = type(value)(self.hi)
        return value


def _infer_kind(default, choices) -> str:
    if choices:
        return "choice"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    raise TypeError(f"cannot tell the parameter kind of {default!r}; pass a Param")


class Filter:
    """A pipeline step with parameters. Subclasses set ``key``, ``title``, ``group`` and
    ``params`` and implement ``apply(frame)``, which returns what a step returns: a Frame,
    a boolean keep-mask or None. Parameter values read like attributes: ``self.radius``.
    """

    key: ClassVar[str] = ""
    title: ClassVar[str] = ""
    group: ClassVar[str] = "Other"
    params: dict[str, Param] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.key:
            REGISTRY[cls.key] = cls

    def __init__(self, **values):
        object.__setattr__(self, "values", {k: p.default for k, p in self.params.items()})
        self.set(**values)

    def set(self, **values) -> None:
        """Change parameter values (checked and clipped to their ranges)."""
        for name, value in values.items():
            if name not in self.params:
                known = ", ".join(self.params) or "none"
                raise TypeError(f"{self.name} has no parameter {name!r} (parameters: {known})")
            self.values[name] = self.params[name].check(value)

    def reset(self) -> None:
        self.values.update({k: p.default for k, p in self.params.items()})

    def __getattr__(self, name):
        values = self.__dict__.get("values")
        if values is not None and name in values:
            return values[name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in self.params:
            self.set(**{name: value})
        else:
            object.__setattr__(self, name, value)

    def __dir__(self):
        return sorted({*super().__dir__(), *self.params})

    @property
    def name(self) -> str:
        """Default step name."""
        return self.key or type(self).__name__.lower()

    @property
    def description(self) -> str:
        return inspect.cleandoc(type(self).__doc__ or "")

    def __call__(self, frame: Frame):
        return self.apply(frame)

    def apply(self, frame: Frame):
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.values.items())
        return f"{type(self).__name__}({args})"


class FunctionFilter(Filter):
    """A filter made from a function ``func(frame, **params)`` (see ``lab.filter``)."""

    def __init__(self, func: Callable, params: dict[str, Param], **values):
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "params", dict(params))
        super().__init__(**values)

    @property
    def name(self) -> str:
        return getattr(self.func, "__name__", "filter")

    @property
    def description(self) -> str:
        return inspect.cleandoc(self.func.__doc__ or "")

    def apply(self, frame: Frame):
        return self.func(frame, **self.values)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.values.items())
        return f"filter {self.name}({args})"


def params_from_signature(func: Callable, specs: dict[str, Any]) -> dict[str, Param]:
    """Parameters of ``func`` after the frame, with defaults; ``specs`` add ranges/choices."""
    params: dict[str, Param] = {}
    signature = inspect.signature(func)
    for k, p in enumerate(signature.parameters.values()):
        if k == 0 or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        spec = specs.get(p.name)
        if isinstance(spec, Param):
            params[p.name] = spec
            continue
        if p.default is p.empty:
            raise TypeError(f"parameter {p.name!r} of {func.__name__} needs a default value")
        params[p.name] = _param_from_spec(p.name, p.default, spec)
    unknown = set(specs) - set(params)
    if unknown:
        raise TypeError(f"{func.__name__} has no parameter(s) {', '.join(sorted(unknown))}")
    return params


def _param_from_spec(name: str, default, spec) -> Param:
    if spec is None:
        return Param(default)
    if isinstance(spec, list) or (isinstance(spec, tuple) and spec and isinstance(spec[0], str)):
        return Param(default, choices=tuple(spec))
    if isinstance(spec, tuple) and len(spec) in (2, 3):
        lo, hi = spec[0], spec[1]
        step = spec[2] if len(spec) == 3 else None
        kind = "int" if isinstance(default, int) and not isinstance(default, bool) else "float"
        return Param(default, lo, hi, step, kind=kind)
    raise TypeError(f"range of {name!r}: use (lo, hi), (lo, hi, step), a list of choices "
                    f"or a Param, not {spec!r}")


def filter_(func: Callable | None = None, *, name: str | None = None, enabled: bool = True,
            to=None, **specs):
    """Decorator: add ``func(frame, **params)`` as a step with parameters (``lab.filter``).

    The parameters are the function's keyword arguments with defaults; give ranges for
    sliders as ``param=(lo, hi)``. Running the script again replaces the step and keeps
    values tuned in the app for parameters whose default did not change.
    """
    from .pipeline import default_pipeline

    def register(f: Callable) -> Callable:
        flt = FunctionFilter(f, params_from_signature(f, specs))
        (default_pipeline() if to is None else to).add(flt, name=name or flt.name,
                                                       enabled=enabled)
        return f

    return register if func is None else register(func)
