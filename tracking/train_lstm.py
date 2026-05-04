"""
Train the LSTM motion predictor on extracted trajectories.

Designed to run on the AWS GPU instance once the quota lands. Local CPU
training is also possible (the model is tiny) but will be slow on a large
dataset.

Workflow:
    # 1. Extract trajectories (one-time per data source)
    python -m tracking.extract_trajectories \
        --gt-files MOT17/train/MOT17-02/gt/gt.txt \
        --out trajectories_mot17.npy

    # 2. Train
    python -m tracking.train_lstm \
        --trajectories trajectories_mot17.npy \
        --out weights/lstm_motion.pt \
        --epochs 30 --batch-size 256

    # 3. Tracker auto-loads weights/lstm_motion.pt on next launch
    python main.py --input demos/slow_walkers.mp4 --classes person

Loss: smooth L1 between predicted and target velocity-delta sequences.
Optimizer: Adam with lr=1e-3, simple constant schedule. The model is small
enough that this just works.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tracking.lstm_motion import (
    LSTMMotionPredictor, MotionDataset,
    DEFAULT_INPUT_LEN, DEFAULT_OUTPUT_LEN,
    DEFAULT_HIDDEN_DIM, DEFAULT_NUM_LAYERS,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    p = argparse.ArgumentParser(description="Train LSTM motion predictor.")
    p.add_argument("--trajectories", type=str, required=True,
                   help="Path to .npy file produced by extract_trajectories.py")
    p.add_argument("--out", type=str, required=True,
                   help="Output checkpoint path (e.g. weights/lstm_motion.pt)")
    p.add_argument("--input-len", type=int, default=DEFAULT_INPUT_LEN)
    p.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    p.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of trajectories held out for validation")
    p.add_argument("--device", type=str, default="auto",
                   help='"auto" | "cpu" | "cuda"')
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Lazy import torch so the rest of the file is at least documentable
    # without it installed.
    import torch
    from torch.utils.data import DataLoader, random_split

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load trajectories
    traj_path = Path(args.trajectories).expanduser().resolve()
    trajectories = np.load(traj_path, allow_pickle=True).tolist()
    total_frames = sum(len(t) for t in trajectories)
    logger.info(
        f"Loaded {len(trajectories)} trajectories ({total_frames} total frames) "
        f"from {traj_path}"
    )

    # Normalize by typical pixel speed so loss values are O(1).
    # Pedestrians move ~2-15 px/frame at 30fps in 1080p; 50 is a generous
    # divisor that keeps inputs roughly in [-1, 1] without clipping fast motion.
    normalize_scale = 50.0

    full_dataset = MotionDataset(
        trajectories,
        input_len=args.input_len,
        output_len=args.output_len,
        normalize_scale=normalize_scale,
    )
    if len(full_dataset) == 0:
        logger.error(
            f"Dataset is empty. Need trajectories with at least "
            f"{args.input_len + args.output_len + 1} consecutive frames each."
        )
        return
    logger.info(f"Dataset windows: {len(full_dataset)}")

    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    # Build model
    model = LSTMMotionPredictor(
        input_len=args.input_len,
        output_len=args.output_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters, device={device}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.SmoothL1Loss()

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n = 0
        for inp, tgt in train_loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            optim.zero_grad()
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optim.step()
            train_loss += loss.item() * inp.size(0)
            n += inp.size(0)
        train_loss /= max(1, n)

        # Val
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp = inp.to(device)
                tgt = tgt.to(device)
                pred = model(inp)
                loss = loss_fn(pred, tgt)
                val_loss += loss.item() * inp.size(0)
                n += inp.size(0)
        val_loss /= max(1, n)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}"
        )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            out_path = Path(args.out).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "input_len": args.input_len,
                    "output_len": args.output_len,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "normalize_scale": normalize_scale,
                },
                "history": history,
                "args": vars(args),
            }, str(out_path))
            logger.info(f"  -> saved best model (val_loss={val_loss:.5f}) to {out_path}")

    logger.info(f"Done. Best val_loss={best_val:.5f}")
    history_path = Path(args.out).with_suffix(".history.json")
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history -> {history_path}")


if __name__ == "__main__":
    main()
