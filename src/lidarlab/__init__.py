"""LidarLab: LiDAR recording viewer with an integrated Python console.

    import lidarlab as lab
    rec = lab.open("data/synthetic/city_rain")
    frame = rec[42]
    frame[frame.echo != 0].intensity
"""

from .core.frame import Frame
from .core.npz import read_frame, write_frame
from .core.pipeline import Pipeline, Step, StepError, pipeline, step
from .core.recording import Recording

__version__ = "0.1.0"

__all__ = [
    "Frame",
    "Pipeline",
    "Recording",
    "Step",
    "StepError",
    "open",
    "pipeline",
    "read_frame",
    "step",
    "write_frame",
]


def open(path, **kwargs) -> Recording:  # noqa: A001 - lab.open(...) reads naturally
    """Open a test run folder: ``rec = lab.open("data/synthetic/city_rain")``."""
    return Recording(path, **kwargs)
