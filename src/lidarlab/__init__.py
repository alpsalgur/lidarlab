"""LidarLab: LiDAR recording viewer with an integrated Python console.

    import lidarlab as lab
    rec = lab.open("data/synthetic/city_rain")
    frame = rec[42]
    frame[frame.echo != 0].intensity

    pipe = lab.Pipeline()
    pipe.add(lab.filters.DROR(beta=3))
    clean = pipe(frame)

    lab.export_run("data/synthetic/city_rain", "data/exports/clean", pipe)
"""

from .core import filters
from .core.batch import ExportReport, export_run
from .core.filter import Filter, Param
from .core.filter import filter_ as filter  # noqa: A004 - lab.filter reads naturally
from .core.formats import write_pcd, write_ply
from .core.frame import Frame
from .core.npz import read_frame, write_frame
from .core.pipeline import Pipeline, Processed, Step, StepError, pipeline, step
from .core.processing import Processor
from .core.recording import Recording

__version__ = "0.2.0"

__all__ = [
    "ExportReport",
    "Filter",
    "Frame",
    "Param",
    "Pipeline",
    "Processed",
    "Processor",
    "Recording",
    "Step",
    "StepError",
    "export_run",
    "filter",
    "filters",
    "open",
    "pipeline",
    "read_frame",
    "step",
    "write_frame",
    "write_pcd",
    "write_ply",
]


def open(path, **kwargs) -> Recording:  # noqa: A001 - lab.open(...) reads naturally
    """Open a test run folder: ``rec = lab.open("data/synthetic/city_rain")``."""
    return Recording(path, **kwargs)
