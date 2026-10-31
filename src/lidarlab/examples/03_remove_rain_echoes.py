"""Remove rain drops and spray in front of real targets, using the echoes.

A drop or a spray cloud gives a weak first echo, and the target behind it the second echo of
the same laser pulse. Dropping such first echoes removes the clutter and keeps the target.
Needs echo-axis data, where ``frame.pulse`` tells which points came from the same pulse.
Tick "Show points removed by the pipeline" under Properties to see what goes.
"""

import numpy as np

import lidarlab as lab

MAX_INTENSITY = 20.0  # the synthetic run stores intensity as 0-255


@lab.step
def remove_rain_echoes(f):
    has_second = np.zeros(f.meta["shape"][1], bool)  # one flag per pulse
    has_second[f.pulse[f.echo == 2]] = True
    rain = (f.echo == 1) & has_second[f.pulse] & (f.intensity < MAX_INTENSITY)
    return ~rain
