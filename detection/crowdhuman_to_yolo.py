"""
CrowdHuman -> YOLO format converter.

CrowdHuman is the standard pretrain dataset for tracking benchmarks (BoT-SORT,
ByteTrack etc. all use it). It contains ~15K crowd-heavy images with dense
person annotations - exactly what off-the-shelf COCO YOLO is weakest on.

CrowdHuman annotation format (ODGT - one JSON per line):
    {"ID": "<image_stem>",
     "gtboxes": [
       {"tag": "person" | "mask",
        "fbox": [x, y, w, h],   # full body box (use this)
        "hbox": [x, y, w, h],   # head box
        "vbox": [x, y, w, h],   # visible body box
        "extra": {"ignore": 0|1, ...},
        "head_attr": {...}},
       ...]}

YOLO training format:
    images/{train,val}/<image_stem>.jpg
    labels/{train,val}/<image_stem>.txt   # one line per box: "0 cx cy w h" (normalized)
    + a YAML describing the dataset

This script converts the official CrowdHuman ODGT annotations into the YOLO
layout that ultralytics' `yolo train` consumes directly.

CLI:
    python -m detection.crowdhuman_to_yolo \
        --crowdhuman-root /path/to/crowdhuman \
        --output-root /path/to/crowdhuman_yolo

Expects on disk:
    crowdhuman/Images/{train,val}/*.jpg   (extracted from CrowdHuman_train0*.zip)
    crowdhuman/annotation_train.odgt
    crowdhuman/annotation_val.odgt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from utils.logger import setup_logger

logger = setup_logger(__name__)


def convert_split(
    images_dir: Path,
    odgt_path: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    skip_ignored: bool = True,
    use_symlinks: bool = True,
) -> tuple[int, int, int]:
    """
    Convert one CrowdHuman split (train or val) to YOLO layout.

    For every image referenced in odgt_path:
      - Symlink (or copy) the .jpg into out_images_dir
      - Write a .txt label file in out_labels_dir with one normalized box per line
      - Use 'fbox' (full body box). Skip boxes flagged as 'ignore' or 'mask'.

    Returns:
        (num_images_processed, num_boxes_written, num_images_missing)
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = n_missing = 0
    with odgt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            stem = rec.get("ID")
            if not stem:
                continue
            src_jpg = images_dir / f"{stem}.jpg"
            if not src_jpg.is_file():
                n_missing += 1
                continue

            # Read image to get dimensions for normalization
            try:
                with Image.open(src_jpg) as im:
                    iw, ih = im.size
            except Exception:
                n_missing += 1
                continue

            label_lines = []
            for box in rec.get("gtboxes", []):
                if box.get("tag") != "person":
                    continue
                if skip_ignored and box.get("extra", {}).get("ignore", 0):
                    continue
                fbox = box.get("fbox")
                if not fbox or len(fbox) != 4:
                    continue
                x, y, w, h = fbox
                # Skip degenerate boxes
                if w <= 0 or h <= 0:
                    continue
                # Clamp to image bounds (CrowdHuman boxes can extend past edges)
                x = max(0.0, float(x))
                y = max(0.0, float(y))
                w = min(float(w), iw - x)
                h = min(float(h), ih - y)
                if w <= 0 or h <= 0:
                    continue
                cx = (x + w / 2) / iw
                cy = (y + h / 2) / ih
                nw = w / iw
                nh = h / ih
                # YOLO needs all values in [0, 1]
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                    continue
                # Class 0 = person (only class in our person-tracking task)
                label_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                n_boxes += 1

            # Write label file (even if empty; YOLO needs .txt to match .jpg)
            (out_labels_dir / f"{stem}.txt").write_text("\n".join(label_lines) + "\n")

            # Place image. Symlinks keep disk usage flat.
            dst_jpg = out_images_dir / f"{stem}.jpg"
            if dst_jpg.exists() or dst_jpg.is_symlink():
                dst_jpg.unlink()
            if use_symlinks:
                try:
                    dst_jpg.symlink_to(src_jpg)
                except OSError:
                    # Fallback to hard link if filesystem restricts symlinks
                    import shutil
                    shutil.copy2(src_jpg, dst_jpg)
            else:
                import shutil
                shutil.copy2(src_jpg, dst_jpg)
            n_images += 1

    return n_images, n_boxes, n_missing


def write_dataset_yaml(out_root: Path) -> Path:
    """Write the data YAML that `yolo train data=...` consumes."""
    yaml_path = out_root / "crowdhuman.yaml"
    yaml_path.write_text(
        f"# Auto-generated by detection/crowdhuman_to_yolo.py\n"
        f"# Single-class person detection. Use this YAML with `yolo train data=...`.\n"
        f"path: {out_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: person\n"
    )
    return yaml_path


def main():
    p = argparse.ArgumentParser(description="Convert CrowdHuman to YOLO format.")
    p.add_argument("--crowdhuman-root", required=True,
                   help="Dir containing Images/{train,val}/*.jpg "
                        "and annotation_{train,val}.odgt")
    p.add_argument("--output-root", required=True,
                   help="Where to write YOLO-format dataset")
    p.add_argument("--include-ignored", action="store_true",
                   help="Include boxes flagged as 'ignore' (default: skip)")
    p.add_argument("--copy", action="store_true",
                   help="Copy images instead of symlinking")
    args = p.parse_args()

    src = Path(args.crowdhuman_root).expanduser().resolve()
    out = Path(args.output_root).expanduser().resolve()

    for split in ("train", "val"):
        images_dir = src / "Images" / split
        odgt_path = src / f"annotation_{split}.odgt"
        if not images_dir.is_dir() or not odgt_path.is_file():
            logger.warning(f"[CrowdHuman] missing {split}: images={images_dir.is_dir()}, "
                           f"odgt={odgt_path.is_file()}")
            continue
        n_imgs, n_boxes, n_missing = convert_split(
            images_dir=images_dir,
            odgt_path=odgt_path,
            out_images_dir=out / "images" / split,
            out_labels_dir=out / "labels" / split,
            skip_ignored=not args.include_ignored,
            use_symlinks=not args.copy,
        )
        logger.info(
            f"[CrowdHuman] {split}: {n_imgs} images, {n_boxes} boxes "
            f"({n_missing} images skipped due to missing files)"
        )

    yaml_path = write_dataset_yaml(out)
    logger.info(f"[CrowdHuman] dataset YAML -> {yaml_path}")
    logger.info(f"[CrowdHuman] ready for: yolo train model=yolo11l.pt data={yaml_path} ...")


if __name__ == "__main__":
    main()
