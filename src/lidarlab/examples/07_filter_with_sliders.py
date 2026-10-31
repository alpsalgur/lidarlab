"""A filter of your own with sliders: drops and spray as range jumps along the scan line.

On one ring (laser) neighbouring pulses of the first echo see the same surface, so their
ranges change smoothly. A rain drop or a spray particle is much closer than both
neighbours. Removing such first echoes is cheap (no KD-tree) and complements DROR.

@lab.filter turns the keyword arguments into parameters; the ranges in the decorator make
them sliders. Select the step in the Pipeline Browser and drag them under Properties.
Needs a ``ring`` field (the synthetic run has one; KITTI does not).
"""

import numpy as np

import lidarlab as lab


@lab.filter(max_jump=(0.1, 5.0, 0.1), max_range=(1.0, 80.0, 1.0), echo=(1, 2))
def ring_jumps(f, max_jump=1.5, max_range=40.0, echo=1):
    """First echoes much closer than both neighbours on their ring (rain drops, spray)."""
    if "ring" not in f.fields:
        raise ValueError("needs a 'ring' field")
    idx = np.flatnonzero(f.echo == echo)
    order = idx[np.lexsort((f.azimuth[idx], f.ring[idx]))]  # scan order: ring, then azimuth
    r = f.range[order]
    ring = f.ring[order]
    step = np.diff(r)  # range change from each point to the next one
    new_ring = ring[1:] != ring[:-1]  # no neighbour across rings: counts as a jump
    closer_than_prev = np.r_[False, (step < -max_jump) | new_ring]
    closer_than_next = np.r_[(step > max_jump) | new_ring, False]
    isolated = closer_than_prev & closer_than_next & (r < max_range)
    keep = np.ones(len(f), bool)
    keep[order[isolated]] = False
    return keep
