"""
DanceTrack -> YOLO format converter.

DanceTrack is a multi-object tracking benchmark of group dance videos. Its
annotations are in MOT-Challenge format (one row per detection per frame, with
a track_id). For training a single-image *detection* model we drop the
track_id and emit YOLO-format per-image labels.

DanceTrack source layout (after extracting train1.zip / train2.zip / val.zip
from the noahcao/dancetrack HF mirror):
    train/<seq>/img1/<NNNNNN>.jpg
    train/<seq>/gt/gt.txt   (MOT format)
    train/<seq>/seqinfo.ini  (provides imWidth, imHeight)
    val/<seq>/...

YOLO training format (output):
    images/{train,val}/<seq>_<frame>.jpg   (symlinked)
    labels/{train,val}/<seq>_<frame>.txt   (one line per box: "0 cx cy w h" normalized)
    + crowdhuman_dancetrack.yaml           (the YAML pointer file for `yolo train`)

We use seq+frame as the unified image stem so label/image names are unique
across all sequences in a split.

CLI:
    python -m detection.dancetrack_to_yolo \
        --dancetrack-root /path/to/dancetrack \
        --output-root /path/to/dancetrack_yolo \
        --merge-with-crowdhuman /path/to/crowdhuman_yolo  # optional

When --merge-with-crowdhuman is set, the resulting YAML lists BOTH datasets
in the train split (CrowdHuman + DanceTrack train) so YOLO sees a combined
training set. This is the standard published-tracker recipe.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)


def _read_seqinfo(seqinfo_path: Path) -> dict:
    """Parse a MOT-style seqinfo.ini for imWidth/imHeight."""
    info = {}
    if not seqinfo_path.is_file():
        return info
    for line in seqinfo_path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def convert_split(
    src_split_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    use_symlinks: bool = True,
) -> tuple[int, int, int]:
    """
    Convert one DanceTrack split (train or val) to YOLO layout.

    Returns:
        (num_sequences, num_images, num_boxes)
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    n_seq = n_img = n_box = 0
    for seq_dir in sorted(src_split_dir.iterdir()):
        if not seq_dir.is_dir():
            continue

        info = _read_seqinfo(seq_dir / "seqinfo.ini")
        try:
            iw = int(info.get("imWidth", 0))
            ih = int(info.get("imHeight", 0))
        except ValueError:
            iw = ih = 0
        if iw <= 0 or ih <= 0:
            logger.warning(f"[DanceTrack] {seq_dir.name}: missing/invalid seqinfo, skipping")
            continue

        gt_path = seq_dir / "gt" / "gt.txt"
        if not gt_path.is_file():
            continue

        # Group all GT boxes by frame
        boxes_by_frame: dict[int, list[tuple]] = {}
        with gt_path.open() as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                try:
                    frame = int(parts[0])
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                except ValueError:
                    continue
                if w <= 0 or h <= 0:
                    continue
                # Clamp to image bounds before normalizing
                x = max(0.0, x)
                y = max(0.0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                if w <= 0 or h <= 0:
                    continue
                cx = (x + w / 2.0) / iw
                cy = (y + h / 2.0) / ih
                nw = w / iw
                nh = h / ih
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                    continue
                boxes_by_frame.setdefault(frame, []).append((cx, cy, nw, nh))

        # Walk the image directory, emit a (label, image) pair per frame
        img_dir = seq_dir / "img1"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            try:
                frame = int(img_path.stem)
            except ValueError:
                continue
            unified_stem = f"{seq_dir.name}_{img_path.stem}"

            # Image (symlink to save disk)
            dst_img = out_images_dir / f"{unified_stem}.jpg"
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            if use_symlinks:
                try:
                    os.symlink(img_path, dst_img)
                except OSError:
                    shutil.copy2(img_path, dst_img)
            else:
                shutil.copy2(img_path, dst_img)

            # Labels (always write, even empty - YOLO expects label file per image)
            label_lines = []
            for cx, cy, nw, nh in boxes_by_frame.get(frame, []):
                # Class 0 = person (DanceTrack is single-class)
                label_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                n_box += 1
            (out_labels_dir / f"{unified_stem}.txt").write_text("\n".join(label_lines) + "\n")
            n_img += 1
        n_seq += 1

    return n_seq, n_img, n_box


def write_dataset_yaml(
    out_root: Path,
    extra_train_dirs: list[Path] | None = None,
) -> Path:
    """
    Write the data YAML that `yolo train data=...` consumes.

    If extra_train_dirs is provided (e.g., CrowdHuman's images/train), the
    training set is the union of DanceTrack train AND those other dirs - this
    is how published trackers combine CrowdHuman + their target benchmark
    train set into one fine-tuning corpus.
    """
    yaml_path = out_root / "dancetrack.yaml"

    train_paths = [str((out_root / "images" / "train").resolve())]
    if extra_train_dirs:
        train_paths.extend(str(Path(p).resolve()) for p in extra_train_dirs)

    if len(train_paths) > 1:
        train_block = "train:\n" + "\n".join(f"  - {p}" for p in train_paths)
    else:
        train_block = f"train: {train_paths[0]}"

    yaml_path.write_text(
        f"# Auto-generated by detection/dancetrack_to_yolo.py\n"
        f"# Single-class person detection. Use with `yolo train data=...`.\n"
        f"path: {out_root.resolve()}\n"
        f"{train_block}\n"
        f"val: {(out_root / 'images' / 'val').resolve()}\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: person\n"
    )
    return yaml_path


def main():
    p = argparse.ArgumentParser(description="Convert DanceTrack to YOLO format.")
    p.add_argument("--dancetrack-root", required=True,
                   help="Directory containing train/, val/ subdirs (after unzipping from HF)")
    p.add_argument("--output-root", required=True,
                   help="Where to write the YOLO-format dataset")
    p.add_argument("--merge-with-crowdhuman", default=None,
                   help="Optional: path to crowdhuman_yolo dir; its images/train will be "
                        "added to the YOLO YAML's train list (combined fine-tuning corpus)")
    p.add_argument("--copy", action="store_true",
                   help="Copy images instead of symlinking")
    args = p.parse_args()

    src = Path(args.dancetrack_root).expanduser().resolve()
    out = Path(args.output_root).expanduser().resolve()

    for split in ("train", "val"):
        split_dir = src / split
        if not split_dir.is_dir():
            logger.warning(f"[DanceTrack] missing split dir: {split_dir}")
            continue
        n_seq, n_img, n_box = convert_split(
            split_dir,
            out / "images" / split,
            out / "labels" / split,
            use_symlinks=not args.copy,
        )
        logger.info(
            f"[DanceTrack] {split}: {n_seq} sequences, {n_img} images, {n_box} boxes"
        )

    extra = []
    if args.merge_with_crowdhuman:
        ch_train = Path(args.merge_with_crowdhuman) / "images" / "train"
        if ch_train.is_dir():
            extra.append(ch_train)
            logger.info(f"[DanceTrack] merging CrowdHuman train into YAML: {ch_train}")
        else:
            logger.warning(f"[DanceTrack] CrowdHuman train dir not found: {ch_train}")

    yaml_path = write_dataset_yaml(out, extra_train_dirs=extra)
    logger.info(f"[DanceTrack] wrote dataset YAML -> {yaml_path}")
    logger.info(f"[DanceTrack] ready: yolo train model=... data={yaml_path} ...")


if __name__ == "__main__":
    main()
