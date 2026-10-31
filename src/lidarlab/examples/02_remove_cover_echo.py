"""Drop echo 0 (the reflection from the sensor cover) from every frame.

Run with F5: the step shows up in the Pipeline Browser and applies while playing. Untick it
there to compare, or tick "Show points removed by the pipeline" under Properties.
"""

import lidarlab as lab


@lab.step
def remove_cover_echo(f):
    return f.echo != 0  # keep-mask: True keeps the point
