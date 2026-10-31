import json
import os
import stat

import pytest

from lidarlab.core.bookmarks import Bookmarks


def test_add_navigate_save_and_load(tmp_path):
    b = Bookmarks(tmp_path)
    b.add(120, " spray behind the truck ")
    b.add(30, "rain starts")
    b.add(120, "spray, heavy")  # same frame: new note
    assert b.frames() == [30, 120] and b.note(120) == "spray, heavy"
    assert b.next(30) == 120 and b.next(120) is None and b.previous(120) == 30
    path = b.save()
    assert path == tmp_path / "bookmarks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [x["frame"] for x in data["bookmarks"]] == [30, 120]
    again = Bookmarks(tmp_path)
    assert again.note(30) == "rain starts" and 120 in again
    again.remove(30)
    assert again.frames() == [120]


@pytest.mark.skipif(os.name != "nt", reason="read-only folders behave differently elsewhere")
def test_read_only_run_folder_uses_the_fallback(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "bookmarks.json").write_text('{"bookmarks": []}')
    os.chmod(run / "bookmarks.json", stat.S_IREAD)  # cannot be rewritten
    try:
        b = Bookmarks(run, fallback_dir=tmp_path / "fallback")
        b.add(5, "x")
        path = b.save()
        assert path.parent == tmp_path / "fallback"
        assert Bookmarks(run, fallback_dir=tmp_path / "fallback").note(5) == "x"  # newest wins
    finally:
        os.chmod(run / "bookmarks.json", stat.S_IWRITE | stat.S_IREAD)
