"""Notes on frames of a test run ("spray behind the truck from here").

Kept in the run's folder as ``bookmarks.json``, so they travel with the data; when that folder
is read-only (e.g. a shared drive), in a fallback folder instead.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

FILE_NAME = "bookmarks.json"


class Bookmarks:
    """Frame -> note for one test run, sorted by frame. ``save()`` after changes."""

    def __init__(self, run_dir, fallback_dir=None):
        self.run_dir = Path(run_dir)
        self.fallback_dir = Path(fallback_dir) if fallback_dir else None
        self.items: dict[int, dict] = {}
        self.path = self.run_dir / FILE_NAME
        # the fallback is only written when the run's file cannot be: the newest one wins,
        # and on a tie (same clock tick) the fallback
        found = [p for p in (self.path, self._fallback_path()) if p is not None and p.is_file()]
        if found:
            run_file = self.path
            self.path = max(found, key=lambda p: (p.stat().st_mtime, p != run_file))
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = {int(b["frame"]): b for b in data.get("bookmarks", [])}

    def _fallback_path(self) -> Path | None:
        if self.fallback_dir is None:
            return None
        key = hashlib.sha1(str(self.run_dir.resolve()).encode()).hexdigest()[:12]
        return self.fallback_dir / f"{self.run_dir.name}_{key}.json"

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, frame) -> bool:
        return int(frame) in self.items

    def frames(self) -> list[int]:
        return sorted(self.items)

    def note(self, frame: int) -> str:
        return self.items.get(int(frame), {}).get("note", "")

    def add(self, frame: int, note: str = "") -> None:
        """Bookmark ``frame`` (an existing bookmark gets the new note)."""
        frame = int(frame)
        old = self.items.get(frame, {})
        self.items[frame] = {"frame": frame, "note": note.strip(),
                             "created": old.get("created") or datetime.now(UTC).isoformat(
                                 timespec="seconds")}

    def remove(self, frame: int) -> None:
        self.items.pop(int(frame), None)

    def next(self, frame: int) -> int | None:
        return next((f for f in self.frames() if f > frame), None)

    def previous(self, frame: int) -> int | None:
        return next((f for f in reversed(self.frames()) if f < frame), None)

    def save(self) -> Path:
        """Write to the run's folder, or to the fallback folder when it is read-only."""
        data = {"bookmarks": [self.items[f] for f in self.frames()]}
        text = json.dumps(data, indent=2, ensure_ascii=False)
        targets = [self.run_dir / FILE_NAME, self._fallback_path()]
        for path in [t for t in targets if t is not None]:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            except OSError:
                continue
            self.path = path
            return path
        raise PermissionError(f"cannot write bookmarks next to {self.run_dir} or elsewhere")
