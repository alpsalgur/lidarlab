import numpy as np
import pytest
from vtkmodules.vtkRenderingCore import vtkCamera

from lidarlab import Frame
from lidarlab.app.picking import BoxSelection, ScreenView, describe_point

WIDTH, HEIGHT = 800, 600


def view(position=(0, 0, 50), focal=(0, 0, 0), up=(0, 1, 0)) -> ScreenView:
    camera = vtkCamera()
    camera.SetPosition(*position)
    camera.SetFocalPoint(*focal)
    camera.SetViewUp(*up)
    camera.SetClippingRange(1, 200)
    m = camera.GetCompositeProjectionTransformMatrix(WIDTH / HEIGHT, -1, 1)
    matrix = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
    return ScreenView(matrix, (0.0, 0.0, float(WIDTH), float(HEIGHT)))


def test_projection_matches_the_camera():
    v = view()
    xy, depth, visible = v.project(np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 60]]))
    np.testing.assert_allclose(xy[0], [WIDTH / 2, HEIGHT / 2], atol=1e-6)
    assert xy[1, 0] > WIDTH / 2 and abs(xy[1, 1] - HEIGHT / 2) < 1e-6  # +x is right
    assert xy[2, 1] > HEIGHT / 2  # +y (view up) is up
    assert visible[:3].all() and not visible[3]  # behind the camera


def test_pick_takes_the_front_point_under_the_mouse():
    v = view()
    pts = np.array([[0, 0, -10], [0, 0, 10], [8, 0, 0]], float)  # two on the same ray
    assert v.pick(pts, WIDTH / 2, HEIGHT / 2) == 1  # z=10 is nearer to the camera at z=50
    assert v.pick(pts, 5, 5) is None
    x, y = v.project(pts[2:])[0][0]
    assert v.pick(pts, x + 3, y - 3) == 2  # within the pixel radius


def test_box_selection_keeps_its_camera():
    v = view()
    grid = np.array([[x, y, 0] for x in range(-10, 11) for y in range(-10, 11)], float)
    box = BoxSelection(v, (WIDTH / 2 + 1, HEIGHT / 2 + 1, WIDTH, HEIGHT))  # upper right
    mask = box.mask(grid)
    assert np.all(grid[mask, 0] > 0) and np.all(grid[mask, 1] > 0)
    assert mask.sum() == 100  # x and y in 1..10
    np.testing.assert_array_equal(box.mask(grid + [0, 0, 0]), mask)  # independent of later views


def test_describe_point():
    f = Frame(np.array([[3.0, 4.0, 0.0], [0, 0, 1]]), {"intensity": np.array([7.0, 1.0])},
              index=np.array([10, 11]))
    p = describe_point(f, 0)
    assert p["range"] == pytest.approx(5.0) and p["azimuth"] == pytest.approx(53.130102, 1e-6)
    assert p["intensity"] == 7.0 and p["index"] == 10
