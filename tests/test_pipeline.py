import numpy as np
import pytest

from lidarlab import Frame, Pipeline, StepError, step


def frame() -> Frame:
    return Frame(np.zeros((4, 3)), {"echo": np.array([0, 1, 2, 1], np.uint8),
                                    "intensity": np.array([1, 2, 3, 4], np.float32)})


def test_steps_return_masks_frames_or_change_fields():
    p = Pipeline()
    p.add(lambda f: f.echo != 0, name="no_cover")
    p.add(lambda f: f[f.intensity > 2], name="bright")

    def scale(f):
        f["intensity"] = f.intensity * 10

    p.add(scale)
    src = frame()
    out = p(src)
    np.testing.assert_array_equal(out.intensity, [30, 40])
    np.testing.assert_array_equal(out.index, [2, 3])
    np.testing.assert_array_equal(src.intensity, [1, 2, 3, 4])  # the input stays unchanged
    assert [(s.n_in, s.n_out) for s in p] == [(4, 3), (3, 2), (2, 2)]
    assert all(s.ms >= 0 for s in p)


def test_same_name_replaces_the_step_in_place():
    p = Pipeline()
    p.add(lambda f: None, name="a")
    p.add(lambda f: None, name="b")
    p.add(lambda f: f.echo == 1, name="a")
    assert [s.name for s in p] == ["a", "b"]
    assert len(p(frame())) == 2


def test_disable_and_reorder():
    p = Pipeline()
    p.add(lambda f: f.echo != 0, name="a")
    p.add(lambda f: f[:1], name="b")
    p.enable("a", False)
    assert len(p(frame())) == 1 and p.active
    p.enable("a")
    p.move("b", -1)
    assert [s.name for s in p] == ["b", "a"]
    assert len(p(frame())) == 0  # b keeps the cover echo point, a removes it


def test_errors_raise_or_skip():
    p = Pipeline()
    p.add(lambda f: 1 / 0, name="broken")
    p.add(lambda f: f.echo != 0, name="ok")
    with pytest.raises(StepError, match="broken"):
        p(frame())
    out = p(frame(), errors="skip")
    assert len(out) == 3
    assert isinstance(p["broken"].error, ZeroDivisionError) and p["ok"].error is None


def test_bad_return_value_is_explained():
    p = Pipeline()
    p.add(lambda f: 5, name="bad")
    with pytest.raises(StepError) as e:
        p(frame())
    assert "boolean mask of 4 values" in str(e.value.__cause__)


def test_decorator_and_listeners():
    p = Pipeline()
    versions = []
    p.subscribe(lambda: versions.append(p.version))

    @step(to=p)
    def keep_first(f):
        return f.echo == 1

    assert "keep_first" in p and versions == [1]
    p.enable("keep_first", False)
    assert versions == [1, 2] and not p.active
    assert p["keep_first"].source[1] > 0  # defined in this file
