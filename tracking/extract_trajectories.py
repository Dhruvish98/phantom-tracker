"""
Extract training trajectories for the LSTM motion predictor.

Two sources supported:
  1. MOT17/MOT20 ground-truth files (gt/gt.txt) - clean, hand-annotated
     trajectories. Best signal but requires the dataset on disk.
  2. Our own tracker output on demo videos - synthetic but free, useful for
     bootstrapping before MOT17 is available.

Each extracted trajectory is a numpy array of shape (T, 2) holding (cx, cy)
center positions in pixel coords. The collection is then handed to
MotionDataset for windowed sampling during training.

CLI:
    python -m tracking.extract_trajectories \
        --gt-files MOT17/train/MOT17-02/gt/gt.txt MOT17/train/MOT17-04/gt/gt.txt \
        --out trajectories_mot17.npy

    python -m tracking.extract_trajectories \
        --videos demos/*.mp4 \
        --out trajectories_demos.npy
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# MOT format extraction
# ─────────────────────────────────────────────────────────────────────

def extract_from_mot_gt(gt_path: Path, min_track_length: int = 32) -> list[np.ndarray]:
    """
    Parse a MOTChallenge gt.txt and return one (T, 2) array per track id.

    MOT format (one detection per line):
        frame, id, x, y, w, h, conf, class, vis, ...

    We keep only person tracks (class==1) with high visibility, and drop
    tracks shorter than min_track_length frames (LSTM needs enough context).
    """
    by_id: dict[int, list[tuple[int, float, float]]] = {}
    with gt_path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                frame = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except ValueError:
                continue
            # Optional MOT17 columns
            cls = int(parts[7]) if len(parts) >= 8 else 1
            vis = float(parts[8]) if len(parts) >= 9 else 1.0
            # Class 1 = pedestrian in MOT17. Visibility 0.3 is the standard
            # MOT-Challenge cutoff for "evaluable" detections.
            if cls != 1 or vis < 0.3:
                continue
            cx = x + w / 2.0
            cy = y + h / 2.0
            by_id.setdefault(track_id, []).append((frame, cx, cy))

    trajectories: list[np.ndarray] = []
    for tid, points in by_id.items():
        # Sort by frame, drop duplicates (rare but possible in some sequences)
        points.sort(key=lambda p: p[0])
        traj = np.array([(cx, cy) for _f, cx, cy in points], dtype=np.float32)
        if len(traj) >= min_track_length:
            trajectories.append(traj)
    logger.info(
        f"[Extract] {gt_path.name}: {len(trajectories)} tracks "
        f">= {min_track_length} frames (from {len(by_id)} total)"
    )
    return trajectories


# ─────────────────────────────────────────────────────────────────────
# Tracker-output extraction (run our own pipeline as the data source)
# ─────────────────────────────────────────────────────────────────────

def extract_from_video(
    video_path: Path,
    yolo_model: str = "yolo11s.pt",
    min_track_length: int = 32,
    classes: list[str] | None = None,
) -> list[np.ndarray]:
    """
    Run the full Phantom Tracker pipeline on a video file and harvest the
    resulting per-track center trajectories. Useful when MOT17 isn't
    available (provides bootstrap data, even if noisier than ground truth).

    Note: imports tracker/detector lazily so this module loads even when
    the heavier deps aren't installed (e.g. quick CLI inspection).
    """
    import cv2
    from core.interfaces import PipelineConfig
    from main import PhantomTracker

    cfg = PipelineConfig(
        yolo_model=yolo_model,
        detect_classes=classes or ["person"],
    )
    pipeline = PhantomTracker(cfg, camera_id=video_path.stem)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"[Extract] Cannot open video: {video_path}")
        return []

    # Collect (cx, cy) per track id, plus track every active id seen
    # (including ones that go OCCLUDED then ACTIVE again in the same session).
    points_by_id: dict[int, list[tuple[float, float]]] = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        state = pipeline.process_frame(frame)
        for t in state.active_tracks:
            cx, cy = t.center
            points_by_id.setdefault(t.track_id, []).append((cx, cy))
    cap.release()

    trajectories: list[np.ndarray] = []
    for tid, pts in points_by_id.items():
        traj = np.array(pts, dtype=np.float32)
        if len(traj) >= min_track_length:
            trajectories.append(traj)
    logger.info(
        f"[Extract] {video_path.name}: {len(trajectories)} tracks "
        f">= {min_track_length} frames (from {len(points_by_id)} total)"
    )
    return trajectories


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Build a trajectory corpus for LSTM training.")
    p.add_argument("--gt-files", nargs="*", default=None,
                   help="One or more MOTChallenge gt.txt paths")
    p.add_argument("--videos", nargs="*", default=None,
                   help="One or more video file paths (will run the full pipeline)")
    p.add_argument("--out", type=str, required=True,
                   help="Output .npy file (saves a list of (T,2) trajectories)")
    p.add_argument("--min-track-length", type=int, default=32,
                   help="Drop tracks shorter than N frames (default 32)")
    p.add_argument("--yolo-model", type=str, default="yolo11s.pt")
    p.add_argument("--classes", type=str, default="person",
                   help="Comma-separated class filter for video extraction")
    args = p.parse_args()

    trajectories: list[np.ndarray] = []

    if args.gt_files:
        for pattern in args.gt_files:
            for path in sorted(glob.glob(pattern)):
                trajectories.extend(extract_from_mot_gt(Path(path), args.min_track_length))

    if args.videos:
        classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
        for pattern in args.videos:
            for path in sorted(glob.glob(pattern)):
                trajectories.extend(extract_from_video(
                    Path(path), args.yolo_model, args.min_track_length, classes
                ))

    if not trajectories:
        logger.error("No trajectories extracted. Provide --gt-files or --videos.")
        return

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.asarray(trajectories, dtype=object), allow_pickle=True)
    total_frames = sum(len(t) for t in trajectories)
    logger.info(
        f"Saved {len(trajectories)} trajectories ({total_frames} total frames) -> {out_path}"
    )


if __name__ == "__main__":
    main()
