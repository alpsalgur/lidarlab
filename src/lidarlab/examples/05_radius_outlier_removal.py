"""Radius outlier removal (ROR) for points closer than MAX_RANGE.

A point with fewer than MIN_NEIGHBOURS other points within RADIUS is dropped: isolated rain
drops have no neighbours, surfaces have many. A fixed radius only suits near points, because
the point spacing grows with range (DROR scales the radius with range). SciPy's KD-tree is
C++, so the Pipeline Browser's "ms" column shows whether this is fast enough to play.
"""

import numpy as np
from scipy.spatial import cKDTree

import lidarlab as lab

RADIUS = 0.4  # m
MIN_NEIGHBOURS = 2
MAX_RANGE = 25.0  # m


@lab.step
def radius_outlier_removal(f):
    near = np.flatnonzero((f.range > 0.5) & (f.range < MAX_RANGE))  # cover echoes stay
    points = f.xyz[near]
    counts = cKDTree(points).query_ball_point(points, RADIUS, return_length=True, workers=-1)
    keep = np.ones(len(f), bool)
    keep[near] = counts > MIN_NEIGHBOURS  # the count includes the point itself
    return keep
