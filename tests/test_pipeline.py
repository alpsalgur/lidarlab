import textwrap

import numpy as np
import pytest

from lidarlab import Frame, Pipeline, StepError, step
from lidarlab.core import filters as F


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


def test_preset_round_trip_with_filters_and_script_steps(tmp_path):
    script = tmp_path / "my_steps.py"
    script.write_text(textwrap.dedent('''
        import lidarlab as lab

        @lab.filter(limit=(0, 10))
        def low_echo(f, limit=1):
            return f.echo <= limit

        @lab.step
        def everything(f):
            return None
    '''))
    p = Pipeline()
    p.add(F.DROR(beta=4.5))
    p.load_dict({"steps": [{"script": str(script), "name": "low_echo"},
                           {"script": str(script), "name": "everything"}]})
    assert [s.name for s in p] == ["low_echo", "everything"]  # load replaces the steps
    p.add(F.EchoSelect(mode="only echo n", n=2), name="second")
    p.move("second", -2)
    p.set("low_echo", limit=2)
    p.enable("everything", False)
    path = p.save(tmp_path / "preset.yaml")

    q = Pipeline()
    versions = []
    q.subscribe(lambda: versions.append(q.version))
    q.load(path)
    assert versions == [q.version]  # one notification for the whole load
    assert [s.name for s in q] == ["second", "low_echo", "everything"]
    assert q["second"].values == {"mode": "only echo n", "n": 2, "cover": 0}
    assert q["low_echo"].values == {"limit": 2}
    assert not q["everything"].enabled
    assert len(q(frame())) == 1  # only echo 2 is left
    assert "# LidarLab pipeline preset" in path.read_text()


def test_steps_defined_in_the_shell_cannot_be_saved():
    p = Pipeline()
    exec("def from_shell(f):\n    return None", ns := {})  # no source file, like the shell
    p.add(ns["from_shell"])
    with pytest.raises(ValueError, match="defined in the shell"):
        p.to_dict()


def test_preset_errors():
    p = Pipeline()
    with pytest.raises(ValueError, match="not a LidarLab pipeline preset"):
        p.load_dict({"name": "x"})
    with pytest.raises(ValueError, match="unknown filter"):
        p.load_dict({"steps": [{"filter": "magic"}]})


def test_filter_steps_show_their_parameters():
    p = Pipeline()
    p.add(F.DROR)  # a class is instantiated with defaults
    s = p["dror"]
    assert s.values["beta"] == 3.0 and "beta" in s.params
    assert "Charron" in s.description
    assert s.source[0].endswith("filters.py")
    assert p.unique_name("dror") == "dror_2" and p.unique_name("sor") == "sor"
    with pytest.raises(TypeError, match="plain function"):
        p.add(lambda f: None, name="plain").set(x=1)
