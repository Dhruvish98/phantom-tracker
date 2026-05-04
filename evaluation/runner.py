"""
Evaluation runner: feed a MOTChallenge sequence through Phantom Tracker and
emit per-frame predictions in the standard MOTChallenge text format. If the
sequence has ground-truth annotations under `gt/gt.txt`, also compute metrics.

CLI:
    python -m evaluation.runner --sequence path/to/MOT17-02 \
                                 --output results/MOT17-02.txt
    python -m evaluation.runner --sequence path/to/MOT17-02 \
                                 --output results/MOT17-02.txt --no-eval

A MOT17-style sequence is a directory containing:
    img1/             - frame images (000001.jpg, 000002.jpg, ...)
    gt/gt.txt         - optional ground-truth annotations (MOT format)
    seqinfo.ini       - optional sequence metadata
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from core.interfaces import PipelineConfig
from main import PhantomTracker
from evaluation.mot_format import write_mot_record
from evaluation.metrics import compute_metrics, format_metrics_table
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Common image extensions in MOTChallenge sequences
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_sequence_frames(seq_dir: Path) -> list[Path]:
    """Locate the image directory inside a MOTChallenge sequence and return sorted frame paths."""
    # Standard layout: <seq>/img1/000001.jpg, ...
    img_dir = seq_dir / "img1"
    if not img_dir.is_dir():
        # Fall back: maybe images live directly under seq_dir
        img_dir = seq_dir
    frames = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTENSIONS
    )
    if not frames:
        raise RuntimeError(
            f"No frames found in {img_dir}. Expected MOTChallenge layout "
            f"(img1/<NNNNNN>.jpg)."
        )
    return frames


def _find_ground_truth(seq_dir: Path) -> Path | None:
    """Look for a gt/gt.txt under the sequence dir; return None if absent."""
    gt_path = seq_dir / "gt" / "gt.txt"
    return gt_path if gt_path.is_file() else None


def run_sequence(
    seq_dir: Path,
    out_path: Path,
    config: PipelineConfig,
    classes: list[str] | None = None,
    progress_every: int = 100,
) -> tuple[int, float]:
    """
    Run the pipeline over every frame in a MOTChallenge sequence and write
    predictions to `out_path` in MOTChallenge format.

    Returns (num_frames_processed, total_seconds).
    """
    if classes:
        config.detect_classes = classes
    config.input_source = str(seq_dir / "img1")  # informational

    frames = _list_sequence_frames(seq_dir)
    logger.info(f"Sequence {seq_dir.name}: {len(frames)} frames")

    pipeline = PhantomTracker(config, camera_id=seq_dir.name)
    records: list[tuple[int, list]] = []
    t_start = time.time()

    for i, frame_path in enumerate(frames, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.warning(f"Failed to read {frame_path}; skipping")
            continue
        state = pipeline.process_frame(frame)
        records.append((i, state.active_tracks))
        if i % progress_every == 0:
            elapsed = time.time() - t_start
            fps = i / elapsed if elapsed > 0 else 0
            logger.info(f"  frame {i}/{len(frames)}  ({fps:.1f} FPS)")

    elapsed = time.time() - t_start
    write_mot_record(out_path, records)
    logger.info(
        f"Wrote {len(records)} frames of predictions to {out_path} "
        f"({elapsed:.1f}s, {len(records)/max(elapsed,1):.1f} FPS)"
    )
    return len(records), elapsed


def main():
    p = argparse.ArgumentParser(description="Run Phantom Tracker on a MOTChallenge sequence and score it.")
    p.add_argument("--sequence", type=str, required=True,
                   help="Path to a MOTChallenge sequence directory (containing img1/)")
    p.add_argument("--output", type=str, required=True,
                   help="Output MOTChallenge predictions file (e.g. results/MOT17-02.txt)")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip metric computation even if ground truth is available")
    p.add_argument("--metrics-out", type=str, default=None,
                   help="Optional: write metrics dict as JSON to this path")
    p.add_argument("--yolo-model", type=str, default="yolo11s.pt")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--classes", type=str, default="person",
                   help="Comma-separated class filter (default: person)")
    p.add_argument("--half", action="store_true")
    args = p.parse_args()

    seq_dir = Path(args.sequence).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    classes = [c.strip() for c in args.classes.split(",")] if args.classes else []

    cfg = PipelineConfig(
        yolo_model=args.yolo_model,
        yolo_confidence=args.confidence,
        detect_classes=classes,
        yolo_half=args.half,
    )

    n_frames, elapsed = run_sequence(seq_dir, out_path, cfg, classes)

    # Score if ground truth available
    gt_path = _find_ground_truth(seq_dir)
    if gt_path and not args.no_eval:
        logger.info(f"Found ground truth at {gt_path}; computing metrics...")
        metrics = compute_metrics(gt_path, out_path)
        metrics["_meta"] = {
            "sequence": seq_dir.name,
            "frames_processed": n_frames,
            "wall_clock_seconds": elapsed,
            "wall_clock_fps": n_frames / max(elapsed, 1),
            # Config snapshot for ablation comparison in the dashboard
            "config": {
                "yolo_model": cfg.yolo_model,
                "yolo_confidence": cfg.yolo_confidence,
                "yolo_half": cfg.yolo_half,
                "detect_classes": cfg.detect_classes,
                "track_high_thresh": cfg.track_high_thresh,
                "track_low_thresh": cfg.track_low_thresh,
                "new_track_thresh": cfg.new_track_thresh,
                "track_buffer": cfg.track_buffer,
                "match_thresh": cfg.match_thresh,
                "appearance_thresh": cfg.appearance_thresh,
                "reid_model": cfg.reid_model,
                "reid_confidence_threshold": cfg.reid_confidence_threshold,
            },
        }
        print()
        print(format_metrics_table(metrics))
        if args.metrics_out:
            mp = Path(args.metrics_out).expanduser().resolve()
            mp.parent.mkdir(parents=True, exist_ok=True)
            with mp.open("w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics written to {mp}")
    elif args.no_eval:
        logger.info("Skipping metric computation (--no-eval)")
    else:
        logger.info(f"No ground truth found at {seq_dir / 'gt' / 'gt.txt'}; predictions only")


if __name__ == "__main__":
    main()
