"""
MOTChallenge text format I/O.

The MOTChallenge benchmark (https://motchallenge.net) defines a plain-text
format used by every published MOT tracker for evaluation. Each line is one
detection/track per frame:

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

  - frame:    1-indexed frame number
  - id:       track identifier (-1 if no association)
  - bb_*:     bounding box in pixel coords (top-left + size)
  - conf:     detection/track confidence (often 1 for ground truth)
  - x, y, z:  3D world coords (-1 when unknown — typical for 2D MOT)

This module:
  - Writes a list of Track snapshots per frame to the MOTChallenge file format
  - Reads MOTChallenge files back into per-frame track dictionaries (used by
    metrics computation against ground-truth annotations)

Used by:
  - evaluation/runner.py     -> writes pipeline output for evaluation
  - evaluation/metrics.py    -> reads gt + predictions for motmetrics
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

from core.interfaces import Track


def write_mot_record(
    out_path: Union[str, Path],
    records: Iterable[tuple[int, list[Track]]],
) -> int:
    """
    Write per-frame track snapshots to MOTChallenge format.

    Args:
        out_path: file path to write to
        records:  iterable of (frame_id, [Track, ...]) pairs in chronological order

    Returns:
        Total number of lines written
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for frame_id, tracks in records:
            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                # MOTChallenge frames are 1-indexed.
                line = (
                    f"{int(frame_id)},{int(t.track_id)},"
                    f"{float(x1):.2f},{float(y1):.2f},{w:.2f},{h:.2f},"
                    f"{float(t.confidence):.4f},-1,-1,-1\n"
                )
                f.write(line)
                n += 1
    return n


def parse_mot_file(path: Union[str, Path]) -> dict[int, list[dict]]:
    """
    Parse a MOTChallenge file into a dict of {frame_id: [{id, bbox, conf}, ...]}.
    Handles both ground-truth files (with extra GT-specific columns ignored)
    and prediction files. Skips malformed lines silently.
    """
    by_frame: dict[int, list[dict]] = {}
    p = Path(path)
    with p.open("r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                frame = int(parts[0])
                obj_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6]) if len(parts) >= 7 else 1.0
            except ValueError:
                continue
            by_frame.setdefault(frame, []).append({
                "id": obj_id,
                "bbox": (x, y, w, h),  # MOT format: x, y, w, h (NOT x1,y1,x2,y2)
                "conf": conf,
            })
    return by_frame
