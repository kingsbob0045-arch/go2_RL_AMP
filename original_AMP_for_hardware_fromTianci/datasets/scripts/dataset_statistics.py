"""
Statistics for all motion dataset txt files.
Duplicate filenames (same name across directories) are assumed to have identical
content and are read only once.

Frame layout (61 values per frame, from AMPLoader):
  [0:3]   root pos    [3:7]   root rot    [7:19]  joint pos
  [19:31] toe pos     [31:34] linear vel  [34:37] angular vel
  [37:49] joint vel   [49:61] toe vel
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict

DATASETS_ROOT = Path(__file__).resolve().parent.parent

# Indices (matches AMPLoader in rsl_rl/datasets/motion_loader.py)
LIN_VEL = slice(31, 34)   # vx, vy, vz
ANG_VEL = slice(34, 37)   # wx, wy, wz


def collect_unique_files(root: Path) -> dict[str, Path]:
    """
    Walk all subdirectories under root and collect txt files.
    When the same filename appears in multiple directories, keep only the first
    occurrence (they are guaranteed to have identical content).
    """
    seen: dict[str, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        if path.name not in seen:
            seen[path.name] = path
    return seen


def parse_motion_file(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)

    frames = data["Frames"]
    n_frames = len(frames)
    frame_dur = float(data["FrameDuration"])

    vx = [fr[31] for fr in frames]
    vy = [fr[32] for fr in frames]
    vz = [fr[33] for fr in frames]
    ang_mag = [math.sqrt(fr[34]**2 + fr[35]**2 + fr[36]**2) for fr in frames]

    return {
        "name": path.name,
        "n_frames": n_frames,
        "fps": round(1.0 / frame_dur, 2) if frame_dur > 0 else 0,
        "motion_weight": float(data.get("MotionWeight", 1.0)),
        "total_duration_s": round(n_frames * frame_dur, 4),
        "vx_min": min(vx), "vx_max": max(vx),
        "vy_min": min(vy), "vy_max": max(vy),
        "vz_min": min(vz), "vz_max": max(vz),
        "ang_min": min(ang_mag), "ang_max": max(ang_mag),
    }


def fmt_range(lo: float, hi: float) -> str:
    return f"[{lo:+.2f}, {hi:+.2f}]"


def print_stats(unique_files: dict[str, Path], all_files: list[Path]) -> None:
    results = [parse_motion_file(p) for p in sorted(unique_files.values(), key=lambda p: p.name)]

    # ── Header ──────────────────────────────────────────────────────────────
    c = "{:<26} {:>6} {:>7} {:>6} {:>9}  {:>16}  {:>16}  {:>16}  {:>16}"
    header = c.format(
        "File", "Frames", "FPS", "Wt", "Dur(s)",
        "vx [min, max]", "vy [min, max]", "vz [min, max]", "‖ω‖ [min, max]",
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    total_dur = 0.0
    for r in results:
        print(c.format(
            r["name"],
            r["n_frames"],
            r["fps"],
            r["motion_weight"],
            r["total_duration_s"],
            fmt_range(r["vx_min"], r["vx_max"]),
            fmt_range(r["vy_min"], r["vy_max"]),
            fmt_range(r["vz_min"], r["vz_max"]),
            fmt_range(r["ang_min"], r["ang_max"]),
        ))
        total_dur += r["total_duration_s"]

    print(sep)

    # ── Summary ─────────────────────────────────────────────────────────────
    n_unique = len(results)
    n_total = len(all_files)
    total_frames = sum(r["n_frames"] for r in results)
    fps_set = sorted({r["fps"] for r in results})

    print(f"\nSummary")
    print(f"  Unique motion files  : {n_unique}  ({n_total - n_unique} duplicates skipped, {n_total} total on disk)")
    print(f"  Total frames         : {total_frames}")
    print(f"  Total duration       : {total_dur:.2f} s  ({total_dur / 60:.2f} min)")
    print(f"  Avg duration / file  : {total_dur / n_unique:.2f} s")
    print(f"  FPS values           : {fps_set}")

    # Global velocity ranges across all files
    print(f"\n  Global velocity ranges across all motions:")
    print(f"    vx : [{min(r['vx_min'] for r in results):+.3f},  {max(r['vx_max'] for r in results):+.3f}]")
    print(f"    vy : [{min(r['vy_min'] for r in results):+.3f},  {max(r['vy_max'] for r in results):+.3f}]")
    print(f"    vz : [{min(r['vz_min'] for r in results):+.3f},  {max(r['vz_max'] for r in results):+.3f}]")
    print(f"    ‖ω‖: [{min(r['ang_min'] for r in results):.3f},  {max(r['ang_max'] for r in results):.3f}]")

    # Per-directory breakdown
    dir_counts: dict[str, int] = defaultdict(int)
    for path in all_files:
        dir_counts[path.parent.name] += 1
    print(f"\n  Per-directory file counts (including duplicates):")
    for d, cnt in sorted(dir_counts.items()):
        print(f"    {d:<40} {cnt:>3} files")


def main():
    all_files = [p for p in sorted(DATASETS_ROOT.rglob("*.txt")) if "scripts" not in p.parts]
    unique_files = {k: v for k, v in collect_unique_files(DATASETS_ROOT).items()
                    if "scripts" not in v.parts}

    print(f"\nDataset root: {DATASETS_ROOT}\n")
    print_stats(unique_files, all_files)


if __name__ == "__main__":
    main()
