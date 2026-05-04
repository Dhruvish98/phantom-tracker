"""
KITTI Tracking -> MOTChallenge format adapter.

Why this module exists:
  Our evaluation/runner.py reads sequences laid out the way MOTChallenge does:
      <seq>/img1/<NNNNNN>.jpg    (1-indexed, 6-digit zero-padded)
      <seq>/gt/gt.txt            (MOTChallenge-format ground truth)
  KITTI Tracking ships its own layout:
      data_tracking_image_2/training/image_02/<NNNN>/<NNNNNN>.png  (0-indexed)
      data_tracking_label_2/training/label_02/<NNNN>.txt           (KITTI format)

  So this module:
   1. Parses KITTI's 17-column annotation format and rewrites it as MOT format.
   2. Adapts the KITTI image directory tree into the MOT layout via symlinks
      (avoids duplicating the ~12GB image set).

KITTI annotation columns (one per detection, space-separated):
  frame, track_id, type, truncated, occluded, alpha,
  bbox_l, bbox_t, bbox_r, bbox_b,
  height, width, length, x, y, z, rotation_y

  type values:  Car | Van | Truck | Pedestrian | Person_sitting | Cyclist
                | Tram | Misc | DontCare

CLI:
  python -m evaluation.kitti_format \
      --kitti-root /path/to/kitti \
      --output-root mot_style/ \
      --sequences 0001 0002 0005

Then evaluation/runner.py treats each `mot_style/<seq>/` as a normal MOT sequence.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)


# Map KITTI class names to "is this class evaluable for tracking?".
# DontCare and Misc are explicitly excluded by KITTI's evaluation protocol.
KITTI_TRACKABLE_CLASSES = {
    "Car", "Van", "Truck", "Pedestrian", "Person_sitting",
    "Cyclist", "Tram",
}
# Common subset for fair comparison with MOT17 (which is pedestrian-only).
KITTI_PEDESTRIAN_CLASSES = {"Pedestrian", "Person_sitting"}


def convert_kitti_label(
    kitti_label_path: Path,
    mot_gt_path: Path,
    classes: set[str] | None = None,
    max_truncation: float = 0.15,
    max_occlusion: int = 1,
) -> int:
    """
    Convert one KITTI label file to MOT-format gt.txt.

    The official KITTI tracking eval protocol restricts evaluation to objects
    that are (a) of an evaluable class, (b) not significantly truncated, and
    (c) at most partly occluded. Including truncated/heavily-occluded objects
    in GT would unfairly count against any 2D detector (the objects are
    physically not visible enough for any detector to find).

    KITTI columns 3 and 4 (0-indexed):
      truncated: float in [0, 1] (fraction of object outside image bounds)
      occluded:  int 0=fully visible, 1=partly, 2=heavily, 3=unknown

    Args:
        kitti_label_path: KITTI's label_02/<seq>.txt
        mot_gt_path:      where to write MOT-format gt.txt
        classes:          KITTI class names to keep (default: trackable classes)
        max_truncation:   drop GT with truncation > this (default 0.15, KITTI default)
        max_occlusion:    drop GT with occlusion > this (default 1, i.e. keep
                          fully-visible and partly-occluded only)

    Returns:
        Number of detection rows written.
    """
    keep = set(classes) if classes else KITTI_TRACKABLE_CLASSES
    n_in = n_kept_class = n_out = 0
    mot_gt_path.parent.mkdir(parents=True, exist_ok=True)
    with kitti_label_path.open() as fin, mot_gt_path.open("w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            n_in += 1
            cls = parts[2]
            if cls not in keep:
                continue
            n_kept_class += 1
            try:
                frame = int(parts[0])
                track_id = int(parts[1])
                truncated = float(parts[3])
                occluded = int(parts[4])
                x1 = float(parts[6])
                y1 = float(parts[7])
                x2 = float(parts[8])
                y2 = float(parts[9])
            except ValueError:
                continue
            # KITTI eval protocol: skip heavily truncated or occluded objects.
            if truncated > max_truncation:
                continue
            if occluded > max_occlusion:
                continue
            # MOT format is 1-indexed for frames; KITTI is 0-indexed.
            mot_frame = frame + 1
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            # MOT GT columns: frame, id, bb_left, bb_top, bb_w, bb_h, conf, class, vis, z
            fout.write(
                f"{mot_frame},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1\n"
            )
            n_out += 1
    logger.info(
        f"[KITTI] {kitti_label_path.name}: {n_in} rows -> {n_kept_class} class-matched -> "
        f"{n_out} after visibility filter (trunc<={max_truncation}, occ<={max_occlusion})"
    )
    return n_out


def prepare_kitti_sequence(
    kitti_root: Path,
    output_root: Path,
    sequence_id: str,
    classes: set[str] | None = None,
    use_symlinks: bool = True,
) -> Path:
    """
    Lay out one KITTI sequence as a MOT-style directory.

    Image frames are symlinked (no copy) by default to save disk space.
    KITTI uses .png, our runner accepts that (we extended _IMG_EXTENSIONS).

    Args:
        kitti_root:   KITTI dataset root (contains data_tracking_image_2/ and
                      data_tracking_label_2/ subdirs)
        output_root:  where to lay out MOT-style sequences
        sequence_id:  4-digit string e.g. "0001"
        classes:      class set to keep in GT (default: all trackable)
        use_symlinks: True symlinks images, False copies

    Returns:
        Path to the prepared sequence directory.
    """
    kitti_root = Path(kitti_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    # Two layouts in the wild:
    #   1. Official multi-zip layout: kitti_root/data_tracking_image_2/training/image_02/<seq>/...
    #   2. Single-zip layout (when each zip is extracted directly into kitti_root):
    #        kitti_root/training/image_02/<seq>/...
    # Probe both and use whichever exists.
    candidates = [
        (kitti_root / "data_tracking_image_2" / "training" / "image_02",
         kitti_root / "data_tracking_label_2" / "training" / "label_02"),
        (kitti_root / "training" / "image_02",
         kitti_root / "training" / "label_02"),
    ]
    src_img_dir = src_label = None
    for img_base, label_base in candidates:
        if (img_base / sequence_id).is_dir() and (label_base / f"{sequence_id}.txt").is_file():
            src_img_dir = img_base / sequence_id
            src_label = label_base / f"{sequence_id}.txt"
            break
    if src_img_dir is None:
        raise FileNotFoundError(
            f"KITTI sequence {sequence_id} not found under {kitti_root}. "
            f"Expected either data_tracking_image_2/training/image_02/{sequence_id}/ "
            f"or training/image_02/{sequence_id}/."
        )

    seq_dir = output_root / sequence_id
    img1_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"
    img1_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Mirror images. KITTI uses 0-indexed 6-digit names (000000.png); MOT uses
    # 1-indexed 6-digit names. Re-number on link.
    src_frames = sorted(src_img_dir.glob("*.png"))
    for src in src_frames:
        try:
            kitti_frame_idx = int(src.stem)
        except ValueError:
            continue
        mot_name = f"{kitti_frame_idx + 1:06d}.png"
        dst = img1_dir / mot_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if use_symlinks:
            try:
                os.symlink(src, dst)
            except OSError:
                # Fall back to copy on Windows without admin / dev mode
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

    # Convert label
    n_rows = convert_kitti_label(src_label, gt_dir / "gt.txt", classes=classes)

    logger.info(
        f"[KITTI] Prepared sequence {sequence_id}: "
        f"{len(src_frames)} frames, {n_rows} GT detections -> {seq_dir}"
    )
    return seq_dir


def main():
    p = argparse.ArgumentParser(
        description="Convert KITTI Tracking sequences to MOT-Challenge layout for evaluation/runner.py"
    )
    p.add_argument("--kitti-root", required=True,
                   help="KITTI root with data_tracking_image_2/ and data_tracking_label_2/")
    p.add_argument("--output-root", required=True,
                   help="Where to lay out MOT-style sequences")
    p.add_argument("--sequences", nargs="+", default=None,
                   help="4-digit sequence IDs (e.g. 0001 0002). Default: all in label_02/")
    p.add_argument("--classes", default="all",
                   help="'all' for all trackable classes, 'pedestrian' for "
                        "Pedestrian+Person_sitting only, or comma-separated list")
    p.add_argument("--copy", action="store_true",
                   help="Copy images instead of symlinking (uses ~12GB extra disk)")
    args = p.parse_args()

    if args.classes == "all":
        classes = KITTI_TRACKABLE_CLASSES
    elif args.classes == "pedestrian":
        classes = KITTI_PEDESTRIAN_CLASSES
    else:
        classes = {c.strip() for c in args.classes.split(",")}

    kitti_root = Path(args.kitti_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if args.sequences:
        seq_ids = args.sequences
    else:
        # Probe both possible label dir layouts (see prepare_kitti_sequence)
        for candidate in [
            kitti_root / "data_tracking_label_2" / "training" / "label_02",
            kitti_root / "training" / "label_02",
        ]:
            if candidate.is_dir():
                seq_ids = [p.stem for p in sorted(candidate.glob("*.txt"))]
                break
        else:
            logger.error(f"No KITTI label dir found under {kitti_root}")
            return

    for sid in seq_ids:
        try:
            prepare_kitti_sequence(kitti_root, output_root, sid, classes=classes,
                                    use_symlinks=not args.copy)
        except FileNotFoundError as e:
            logger.warning(f"  skipped {sid}: {e}")


if __name__ == "__main__":
    main()
