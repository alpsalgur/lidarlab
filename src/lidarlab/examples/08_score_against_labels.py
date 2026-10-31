"""Score the pipeline against ground-truth labels: how much of each noise class it removes,
and how much of the real scene goes with it.

Needs a test run with a ``label`` field and ``noise_labels`` in its recording.json: the
synthetic run (cover 0, rain 5, spray 6) or WADS (falling snow). Load a preset or add
filters first, then run this (F5) on a few frames. Noise the pipeline missed is painted
magenta on every frame; color by "label" to see all classes.
"""

import numpy as np

NOISE = rec.info.get("noise_labels", [])
NAMES = rec.info.get("labels", {})
if "label" not in view.frame.fields or not NOISE:
    raise SystemExit("this test run has no ground-truth labels")

raw, out = view.frame, view.output
removed = ~np.isin(raw.index, out.index)
noise = np.isin(raw.label, NOISE)
print(f"frame {view.index}: {len(raw):,} points, {noise.sum():,} of them noise")
for c in NOISE:
    is_c = raw.label == c
    if is_c.any():
        share = np.count_nonzero(removed & is_c) / is_c.sum()
        print(f"  {NAMES.get(str(c), c)!s:>14}: {share:6.1%} of {is_c.sum():,} removed")
lost = np.count_nonzero(removed & ~noise)
print(f"  {'real points':>14}: {lost:,} removed by mistake "
      f"({lost / max((~noise).sum(), 1):.2%} of the scene)")
print(f"  precision: {np.count_nonzero(removed & noise) / max(removed.sum(), 1):.1%} of the "
      "removed points were noise; missed noise is magenta")


@view.highlight(color="magenta", size=4)
def missed_noise(f):
    return np.isin(f.label, NOISE)
