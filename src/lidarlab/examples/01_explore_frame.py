"""Look around the current frame: fields and value ranges, echoes, pose.

Uses whatever frame is shown: step to another frame (→) and press F5 again.
"""

import numpy as np

print(frame)
for name, values in frame.fields.items():
    lo, hi = np.nanmin(values), np.nanmax(values)
    print(f"  {name:>10}  {str(values.dtype):>8}  {lo:10.4g} … {hi:.4g}")

if "echo" in frame:
    print("\necho     points   median range (m)   median intensity")
    for e in np.unique(frame.echo):
        part = frame[frame.echo == e]
        intensity = np.median(part.intensity) if "intensity" in part else float("nan")
        print(f"{e:>4} {len(part):>10,} {np.median(part.range):18.2f} {intensity:18.2f}")

pose = {k: round(float(np.degrees(frame.meta[k])), 3)
        for k in ("roll", "pitch", "yaw") if k in frame.meta}
print("\npose (deg):", pose)
