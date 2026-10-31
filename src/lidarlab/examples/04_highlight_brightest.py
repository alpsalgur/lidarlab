"""Paint the brightest returns yellow, without removing anything.

Retro-reflective signs, license plates and lane markings stand out. The threshold is a
percentile of the frame, so this works on any test run whatever its intensity scale.
A highlight is recomputed for every frame, like a step, but only draws on top.
"""

import numpy as np

TOP_PERCENT = 0.5


@view.highlight(color="yellow", size=4)
def brightest(f):
    return f.intensity > np.percentile(f.intensity, 100 - TOP_PERCENT)
