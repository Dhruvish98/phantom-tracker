"""
MOT metric computation via the official `motmetrics` library.

Computes the standard MOT-Challenge metrics:
  - MOTA (Multi-Object Tracking Accuracy): combined error rate (FN + FP + IDsw)
  - MOTP (Multi-Object Tracking Precision): localization accuracy
  - IDF1: ID-aware F1 score, weights long-term identity preservation
  - IDP / IDR: ID precision / recall
  - MT / ML: number of mostly-tracked / mostly-lost ground-truth tracks
  - num_switches (IDsw): identity switch count
  - num_fragmentations (FM): track fragmentation count
  - precision / recall

Usage:
    metrics = compute_metrics(
        gt_path="MOT17/train/MOT17-02/gt/gt.txt",
        pred_path="results/MOT17-02.txt",
    )
    print(metrics["MOTA"], metrics["IDF1"])

The function delegates the heavy lifting to motmetrics' MOTAccumulator and
mh.compute(). We just frame-pair the data and feed the (gt_id, pred_id, IoU)
triples it expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

# NumPy 2.0 removed np.asfarray; motmetrics 1.4.0 still calls it. Polyfill
# before importing motmetrics so its iou_matrix() keeps working.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

try:
    import motmetrics as mm
except ImportError:
    mm = None

from evaluation.mot_format import parse_mot_file
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _xywh_to_iou_input(boxes: list[tuple]) -> np.ndarray:
    """motmetrics expects boxes as (X, Y, W, H) - same as MOT format already."""
    return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)


def compute_metrics(
    gt_path: Union[str, Path],
    pred_path: Union[str, Path],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute MOTChallenge metrics for one prediction file vs one ground-truth file.

    Args:
        gt_path:        path to ground-truth file (MOTChallenge format)
        pred_path:      path to prediction file (MOTChallenge format)
        iou_threshold:  IoU below which a detection is considered unmatched

    Returns:
        dict of metric_name -> value
    """
    if mm is None:
        raise RuntimeError(
            "motmetrics not installed. Install with: pip install motmetrics"
        )

    gt = parse_mot_file(gt_path)
    pred = parse_mot_file(pred_path)
    all_frames = sorted(set(gt.keys()) | set(pred.keys()))

    acc = mm.MOTAccumulator(auto_id=False)
    for frame in all_frames:
        gt_objs = gt.get(frame, [])
        pred_objs = pred.get(frame, [])

        gt_ids = [o["id"] for o in gt_objs]
        pred_ids = [o["id"] for o in pred_objs]
        gt_boxes = _xywh_to_iou_input([o["bbox"] for o in gt_objs])
        pred_boxes = _xywh_to_iou_input([o["bbox"] for o in pred_objs])

        # IoU distance matrix (motmetrics convention: distance = 1 - IoU,
        # entries above max_iou are masked as nan -> not matched)
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            distance = mm.distances.iou_matrix(
                gt_boxes, pred_boxes, max_iou=1.0 - iou_threshold,
            )
        else:
            distance = np.empty((len(gt_boxes), len(pred_boxes)))

        acc.update(gt_ids, pred_ids, distance, frameid=frame)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames", "num_objects", "num_predictions",
            "num_matches", "num_misses", "num_false_positives",
            "num_switches", "num_fragmentations",
            "mota", "motp", "idf1", "idp", "idr",
            "precision", "recall", "mostly_tracked", "mostly_lost",
        ],
        name=str(Path(pred_path).stem),
    )

    # Convert single-row DataFrame into a flat dict.
    row = summary.iloc[0].to_dict()
    # motmetrics returns lowercase keys; canonicalize to the names typically
    # cited in MOT papers (uppercase MOTA/MOTP/IDF1/etc.)
    canonical = {
        "MOTA": row["mota"], "MOTP": row["motp"],
        "IDF1": row["idf1"], "IDP": row["idp"], "IDR": row["idr"],
        "Precision": row["precision"], "Recall": row["recall"],
        "MT": int(row["mostly_tracked"]),
        "ML": int(row["mostly_lost"]),
        "IDSwitches": int(row["num_switches"]),
        "Fragmentations": int(row["num_fragmentations"]),
        "FN": int(row["num_misses"]),
        "FP": int(row["num_false_positives"]),
        "TP": int(row["num_matches"]),
        "GTObjects": int(row["num_objects"]),
        "Predictions": int(row["num_predictions"]),
        "Frames": int(row["num_frames"]),
    }
    return canonical


def format_metrics_table(metrics: dict) -> str:
    """Pretty-print a metrics dict to a one-screen text table."""
    lines = ["=" * 50]
    lines.append("MOT METRICS")
    lines.append("=" * 50)
    # Headline metrics
    headline = ["MOTA", "MOTP", "IDF1", "IDP", "IDR"]
    for k in headline:
        v = metrics.get(k)
        if v is not None:
            lines.append(f"  {k:<6}  {v:>8.3f}")
    lines.append("-" * 50)
    # Counts
    counts = [
        ("MT", "mostly tracked"),
        ("ML", "mostly lost"),
        ("IDSwitches", "ID switches"),
        ("Fragmentations", "track fragmentations"),
        ("FN", "false negatives (misses)"),
        ("FP", "false positives"),
        ("TP", "true positives"),
        ("GTObjects", "ground-truth detections"),
        ("Predictions", "predicted detections"),
        ("Frames", "frames evaluated"),
    ]
    for k, desc in counts:
        v = metrics.get(k)
        if v is not None:
            lines.append(f"  {k:<16}  {v:>8}   {desc}")
    lines.append("=" * 50)
    return "\n".join(lines)
