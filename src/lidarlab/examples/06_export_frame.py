"""Save the current frame after the processing steps, in the layout of the source file.

In echo-axis files the removed points become NaN, so the result loads like the original.
The file goes to data/exports/<test run>/, which the Test Runs panel lists after a rescan.
"""

out_dir = rec.path.parents[1] / "exports" / rec.name
path = view.output.save(out_dir / f"frame_{view.index:06d}.npz")
print(f"saved {len(view.output):,} of {len(view.frame):,} points to {path}")
