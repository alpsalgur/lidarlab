"""Run a pipeline over a whole test run (or part of it) and write the frames.

    report = lab.export_run("data/synthetic/city_rain", "data/exports/city_rain_clean",
                            pipeline="src/lidarlab/examples/presets/rain_denoise.yaml")
    print(report.summary())

    lidarlab run data/synthetic/city_rain data/exports/clean --pipeline rain_denoise.yaml

The output folder is a test run again: one file per source frame under the same name, in the
source's npz layout (removed points become NaN in echo-axis files) or as PLY/PCD, plus
``recording.json`` (what was done), ``pipeline.yaml`` (the steps) and ``export.log`` (points in
and out, time and error per frame). A frame whose steps fail is not written: an unfiltered
frame must not slip into the output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .formats import write_pcd, write_ply
from .npz import read_frame, write_frame
from .pipeline import Pipeline
from .recording import Recording, natural_key

FORMATS = {"npz": ".npz", "ply": ".ply", "pcd": ".pcd"}
Progress = Callable[[int, int, "FrameResult"], "bool | None"]  # False cancels


@dataclass
class FrameResult:
    index: int
    name: str
    n_in: int = 0
    n_out: int = 0
    ms: float = 0.0
    error: str | None = None


@dataclass
class ExportReport:
    out_dir: Path
    total: int
    results: list[FrameResult] = field(default_factory=list)
    seconds: float = 0.0
    cancelled: bool = False

    @property
    def failed(self) -> list[FrameResult]:
        return [r for r in self.results if r.error is not None]

    @property
    def written(self) -> int:
        return len(self.results) - len(self.failed)

    def summary(self) -> str:
        rate = len(self.results) / max(self.seconds, 1e-9)
        text = (f"{self.written}/{self.total} frames written to {self.out_dir} in "
                f"{self.seconds:.1f} s ({rate:.1f} frames/s)")
        if self.failed:
            text += f"; {len(self.failed)} failed, see export.log"
        if self.cancelled:
            text += "; cancelled"
        return text


def default_workers() -> int:
    """All logical cores but two, at most 8: 6 threads exported fastest on 4 cores / 8 threads
    (threads x4 5.4, x6 6.6, x8 5.3 frames/s; processes x6 5.6)."""
    return max(1, min(8, (os.cpu_count() or 4) - 2))


def select_frames(frames, count: int) -> list[int]:
    """Frame numbers from None (all), "A:B" (A included, B not; either may be left out),
    a range/slice or any iterable of ints. Negative numbers count from the end."""
    if frames is None:
        return list(range(count))
    if isinstance(frames, str):
        start, sep, stop = frames.partition(":")
        if not sep:
            raise ValueError(f"frames must look like A:B, got {frames!r}")
        frames = slice(int(start) if start.strip() else None, int(stop) if stop.strip() else None)
    if isinstance(frames, slice):
        return list(range(count))[frames]
    picked = []
    for i in frames:
        i = int(i) + (count if int(i) < 0 else 0)
        if not 0 <= i < count:
            raise IndexError(f"frame {i} out of range 0-{count - 1}")
        picked.append(i)
    return picked


def _source(source) -> tuple[list[Path], dict, Path]:
    if isinstance(source, Recording):
        return list(source.files), dict(source.info), source.path
    path = Path(source)
    if not path.is_dir():
        raise NotADirectoryError(f"not a folder: {path}")
    files = sorted(path.glob("*.npz"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"no .npz files in {path}")
    info_file = path / "recording.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    return files, info, path


def _resolve_pipeline(pipeline) -> Pipeline:
    if pipeline is None:
        return Pipeline()
    if isinstance(pipeline, Pipeline):
        return pipeline.snapshot()  # the caller may keep editing theirs meanwhile
    loaded = Pipeline()
    loaded.load(pipeline)
    return loaded


def _clear_export(out_dir: Path) -> None:
    """Remove an earlier export's frames and files, so no stale frame stays in the folder;
    other files and subfolders are left alone."""
    names = {"recording.json", "pipeline.yaml", "export.log"}
    for path in out_dir.iterdir():
        if path.is_file() and (path.suffix in FORMATS.values() or path.name in names):
            path.unlink()


def _write(frame, path: Path, fmt: str, compress: bool) -> None:
    if fmt == "npz":
        write_frame(frame, path, compress=compress)
    elif fmt == "ply":
        write_ply(frame, path)
    else:
        write_pcd(frame, path)


def process_file(pipeline: Pipeline, src: Path, index: int, dst: Path, fmt: str = "npz",
                 compress: bool = True) -> FrameResult:
    """Read, process and write one frame; errors are returned, not raised."""
    start = time.perf_counter()
    result = FrameResult(index, src.name)
    try:
        raw = read_frame(src, frame_no=index)
        result.n_in = len(raw)
        out = pipeline.process(raw, errors="raise").frame
        result.n_out = len(out)
        _write(out, dst, fmt, compress)
    except Exception as e:  # noqa: BLE001 - reported per frame
        result.error = f"{type(e).__name__}: {e}"
    result.ms = (time.perf_counter() - start) * 1e3
    return result


_worker_pipeline: Pipeline | None = None  # per worker process


def _init_process(preset: dict) -> None:
    global _worker_pipeline
    _worker_pipeline = Pipeline()
    _worker_pipeline.load_dict(preset)


def _process_in_worker(src, index, dst, fmt, compress) -> FrameResult:
    return process_file(_worker_pipeline, src, index, dst, fmt, compress)


def export_run(source, out_dir, pipeline=None, frames=None, fmt: str = "npz",
               workers: int | None = None, processes: bool = False, compress: bool = True,
               overwrite: bool = False, progress: Progress | None = None) -> ExportReport:
    """Run ``pipeline`` (a Pipeline, a preset file or None) over the frames of ``source``
    (a Recording or a folder) and write them to ``out_dir``.

    ``workers`` threads share the work (NumPy, SciPy and zlib release the GIL); with
    ``processes`` they are processes, which needs a pipeline that fits in a preset. On
    Windows, a script using processes must call this under ``if __name__ == "__main__":``
    (each worker imports the script again).
    ``progress(done, total, result)`` is called after every frame; returning False cancels:
    frames in progress finish, the rest are skipped.
    """
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}, not {fmt!r}")
    files, info, src_dir = _source(source)
    indices = select_frames(frames, len(files))
    out_dir = Path(out_dir)
    if out_dir.resolve() == src_dir.resolve():
        raise ValueError("the output folder is the test run itself")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty (overwrite=True replaces its files)")
        _clear_export(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = _resolve_pipeline(pipeline)
    try:
        preset = pipe.to_dict()
    except ValueError:
        preset = None  # steps typed into the shell: no preset, threads only
    if processes and preset is None:
        raise ValueError("worker processes need a pipeline that can be saved as a preset; "
                         "define the steps in a script file, or use threads")
    workers = workers or default_workers()
    report = ExportReport(out_dir, len(indices))
    jobs = iter([(files[i], i, out_dir / (files[i].stem + FORMATS[fmt]), fmt, compress)
                 for i in indices])
    executor: Executor
    if processes:
        executor = ProcessPoolExecutor(workers, initializer=_init_process, initargs=(preset,))
    else:
        executor = ThreadPoolExecutor(workers, thread_name_prefix="lidarlab-export")
    start = time.perf_counter()
    pending = set()

    def submit_more() -> None:
        while len(pending) < 2 * workers:  # bounded: memory stays flat, cancel stays quick
            job = next(jobs, None)
            if job is None:
                return
            if processes:
                pending.add(executor.submit(_process_in_worker, *job))
            else:
                pending.add(executor.submit(process_file, pipe, *job))

    with executor:
        try:
            submit_more()
            while pending:
                done, still = wait(pending, return_when=FIRST_COMPLETED)
                pending.clear()
                pending.update(still)
                for future in done:
                    result = future.result()
                    report.results.append(result)
                    if progress is not None and progress(len(report.results), report.total,
                                                         result) is False:
                        report.cancelled = True
                if report.cancelled:
                    break
                submit_more()
        except KeyboardInterrupt:
            report.cancelled = True
        if report.cancelled:
            for future in pending:
                future.cancel()
            finished, _ = wait(pending)
            report.results.extend(f.result() for f in finished if not f.cancelled())
    report.seconds = time.perf_counter() - start
    report.results.sort(key=lambda r: r.index)
    _write_metadata(out_dir, info, src_dir, pipe, preset, report, fmt)
    return report


def _write_metadata(out_dir: Path, info: dict, src_dir: Path, pipe: Pipeline,
                    preset: dict | None, report: ExportReport, fmt: str) -> None:
    from lidarlab import __version__

    done = [r.index for r in report.results]
    info = dict(info)
    info["processed"] = {
        "source": str(src_dir.resolve()),
        "frames": f"{min(done)}-{max(done)}" if done else "none",
        "format": fmt,
        "steps": [s.name for s in pipe if s.enabled],
        "pipeline": preset,
        "written": report.written,
        "failed": len(report.failed),
        "cancelled": report.cancelled,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "lidarlab": __version__,
    }
    (out_dir / "recording.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if preset is not None and len(pipe):
        pipe.save(out_dir / "pipeline.yaml")
    lines = ["frame\tfile\tpoints_in\tpoints_out\tms\terror"]
    lines += [f"{r.index}\t{r.name}\t{r.n_in}\t{r.n_out}\t{r.ms:.1f}\t{r.error or ''}"
              for r in report.results]
    (out_dir / "export.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _progress_line(start: float) -> Progress:
    def show(done: int, total: int, result: FrameResult) -> None:
        elapsed = time.perf_counter() - start
        rate = done / max(elapsed, 1e-9)
        left = (total - done) / max(rate, 1e-9)
        bar = "#" * int(30 * done / max(total, 1))
        failed = f", last error: {result.error}" if result.error else ""
        sys.stdout.write(f"\r[{bar:<30}] {done}/{total} frames, {rate:.1f}/s, "
                         f"{int(left) // 60}:{int(left) % 60:02d} left{failed[:60]}   ")
        sys.stdout.flush()

    return show


def main(argv: Iterable[str] | None = None) -> int:
    """``lidarlab run``: the same processing as the app's Export, without a window."""
    ap = argparse.ArgumentParser(
        prog="lidarlab run",
        description="Run a pipeline preset over a test run and write the processed frames.")
    ap.add_argument("source", type=Path, help="test run folder (.npz frames)")
    ap.add_argument("out", type=Path, help="output folder (becomes a test run)")
    ap.add_argument("-p", "--pipeline", type=Path,
                    help="preset (.yaml); without it the frames are only copied or converted")
    ap.add_argument("--frames", help="A:B, first included, last not (e.g. 100:300 or 400:)")
    ap.add_argument("--format", choices=list(FORMATS), default="npz")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="threads or processes (default: %(default)s)")
    ap.add_argument("--processes", action="store_true", help="use processes instead of threads")
    ap.add_argument("--no-compress", action="store_true", help="plain .npz: bigger, faster")
    ap.add_argument("--overwrite", action="store_true", help="write into a non-empty folder")
    a = ap.parse_args(list(argv) if argv is not None else None)
    try:
        report = export_run(a.source, a.out, a.pipeline, a.frames, a.format, a.workers,
                            a.processes, not a.no_compress, a.overwrite,
                            progress=_progress_line(time.perf_counter()))
    except (OSError, ValueError, KeyError) as e:
        print(f"lidarlab run: {e}", file=sys.stderr)
        return 2
    print()
    print(report.summary())
    for r in report.failed[:5]:
        print(f"  frame {r.index} ({r.name}): {r.error}")
    if report.cancelled:
        return 130
    return 1 if report.failed else 0
