"""Picking and box selection in screen space.

Points are projected with the camera's own matrix, exactly where VTK draws them, so a click
picks what is under the mouse and a box selects what is inside it. Coordinates are VTK display
pixels (origin bottom-left, physical pixels).

A box selection keeps the camera it was drawn with, so it stays fixed relative to the sensor
and can be applied to every frame of a test run (e.g. "points in the spray region over time").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScreenView:
    """A camera's world-to-display mapping: a 4x4 matrix to clip space (VTK's composite
    projection) and the viewport (x0, y0, width, height) in display pixels."""

    matrix: np.ndarray
    viewport: tuple[float, float, float, float]

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Display positions (N, 2), depth (-1 near … 1 far) and which points are in front of
        the camera and between its clipping planes."""
        m = self.matrix
        xyz = np.asarray(xyz, np.float64)
        clip = xyz @ m[:3, :3].T + m[:3, 3]
        w = xyz @ m[3, :3] + m[3, 3]
        visible = w > 1e-12
        w = np.where(visible, w, 1.0)
        ndc = clip / w[:, None]
        visible &= np.abs(ndc[:, 2]) <= 1.0
        x0, y0, width, height = self.viewport
        xy = np.c_[x0 + (ndc[:, 0] + 1.0) * 0.5 * width, y0 + (ndc[:, 1] + 1.0) * 0.5 * height]
        return xy, ndc[:, 2], visible

    def pick(self, xyz: np.ndarray, x: float, y: float, radius: float = 8.0) -> int | None:
        """Index of the point under (x, y): the nearest to the camera among the points drawn
        within ``radius`` pixels, or None."""
        if not len(xyz):
            return None
        xy, depth, visible = self.project(xyz)
        near = visible & (((xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2) <= radius * radius)
        candidates = np.flatnonzero(near)
        if not candidates.size:
            return None
        return int(candidates[np.argmin(depth[candidates])])

    def inside(self, xyz: np.ndarray, rect) -> np.ndarray:
        """Mask: points drawn inside rect = (x0, y0, x1, y1), corners in any order."""
        if not len(xyz):
            return np.zeros(0, bool)
        xy, _, visible = self.project(xyz)
        x0, x1 = sorted((rect[0], rect[2]))
        y0, y1 = sorted((rect[1], rect[3]))
        return visible & (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)


def screen_view(renderer) -> ScreenView:
    """The current mapping of a VTK renderer."""
    width, height = renderer.GetSize()
    x0, y0 = renderer.GetOrigin()
    camera = renderer.GetActiveCamera()
    m = camera.GetCompositeProjectionTransformMatrix(width / max(height, 1), -1.0, 1.0)
    matrix = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
    return ScreenView(matrix, (float(x0), float(y0), float(width), float(height)))


@dataclass(frozen=True)
class BoxSelection:
    """A screen rectangle with the camera it was drawn with; applies to any frame."""

    view: ScreenView
    rect: tuple[float, float, float, float]

    def mask(self, xyz: np.ndarray) -> np.ndarray:
        return self.view.inside(xyz, self.rect)


def describe_point(frame, i: int) -> dict:
    """All fields of point ``i`` plus range, azimuth and elevation."""
    point = frame[int(i)]
    x, y, z = point["x"], point["y"], point["z"]
    point["range"] = float(np.sqrt(x * x + y * y + z * z))
    point["azimuth"] = float(np.degrees(np.arctan2(y, x)))
    point["elevation"] = float(np.degrees(np.arctan2(z, np.hypot(x, y))))
    point["index"] = int(frame.index[int(i)])
    return point
