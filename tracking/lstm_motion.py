"""
LSTM-based motion predictor for occluded tracks.

Replaces the linear velocity extrapolation in tracking/tracker.py with a
learned sequence model. Handles non-linear motion (turns, accel/decel) where
linear extrapolation drifts off the true path.

Design:
  - Input:  N past frame-to-frame velocity deltas (dx, dy), shape (B, N, 2)
  - Output: K future velocity deltas, shape (B, K, 2)
  - Inference is a single LSTM forward pass + linear projection (not
    autoregressive) - simpler, faster, plenty of expressivity at this scale.
  - Velocity-space (not position) is translation-invariant; the model learns
    motion patterns regardless of where on the frame the track is.

Public API:
  - LSTMMotionPredictor: nn.Module, encapsulates the model
  - MotionDataset: torch.utils.data.Dataset for training, slices trajectories
    into (input, target) pairs
  - load_predictor(weights_path): convenience loader for the tracker; returns
    None if file missing or torch unavailable
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    Dataset = object
    _TORCH_AVAILABLE = False

from utils.logger import setup_logger

logger = setup_logger(__name__)


# Defaults sized for our use case:
#   30 FPS video, occlusion buffer 90 frames (3s) -> 15 frames lookahead is plenty.
#   Input window of 15 deltas (16 positions = ~0.5s of past) gives the model
#   enough context to capture acceleration/curvature without being wasteful.
DEFAULT_INPUT_LEN = 15
DEFAULT_OUTPUT_LEN = 15
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 1


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class LSTMMotionPredictor(nn.Module):
        """
        Encode -> single-shot decode LSTM motion predictor.
        ~20K params with defaults; CPU-friendly per-frame inference.
        """
        def __init__(
            self,
            input_len: int = DEFAULT_INPUT_LEN,
            output_len: int = DEFAULT_OUTPUT_LEN,
            hidden_dim: int = DEFAULT_HIDDEN_DIM,
            num_layers: int = DEFAULT_NUM_LAYERS,
        ):
            super().__init__()
            self.input_len = input_len
            self.output_len = output_len
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.encoder = nn.LSTM(
                input_size=2, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
            )
            # Project final hidden state to all K future deltas at once.
            self.head = nn.Linear(hidden_dim, output_len * 2)

        def forward(self, deltas: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                deltas: (B, T, 2) past velocity deltas (T=input_len typically)
            Returns:
                (B, output_len, 2) predicted future velocity deltas
            """
            _, (h_n, _) = self.encoder(deltas)
            last_hidden = h_n[-1]                           # (B, hidden_dim)
            flat = self.head(last_hidden)                   # (B, output_len * 2)
            return flat.view(-1, self.output_len, 2)


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class MotionDataset(Dataset):
    """
    Training dataset built from raw position trajectories.

    Each sample is a (past_deltas, future_deltas) pair sliced from one
    trajectory. We slide a window over each trajectory to produce many
    training samples per track.

    Input trajectories should be lists of np.ndarray of shape (T, 2)
    holding (cx, cy) center positions in pixel coords.
    """

    def __init__(
        self,
        trajectories: Iterable[np.ndarray],
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        stride: int = 1,
        normalize_scale: float = 1.0,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch not available; cannot build MotionDataset")
        self.input_len = input_len
        self.output_len = output_len
        self.normalize_scale = normalize_scale

        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        # We need (input_len + output_len + 1) consecutive positions to produce
        # one sample (we differentiate to get input_len + output_len deltas).
        window_positions = input_len + output_len + 1

        for traj in trajectories:
            traj = np.asarray(traj, dtype=np.float32)
            if traj.ndim != 2 or traj.shape[1] != 2:
                continue
            if len(traj) < window_positions:
                continue
            # Compute all deltas once, then slice
            deltas = np.diff(traj, axis=0)   # (T-1, 2)
            for start in range(0, len(deltas) - input_len - output_len + 1, stride):
                inp = deltas[start : start + input_len]
                tgt = deltas[start + input_len : start + input_len + output_len]
                if len(inp) == input_len and len(tgt) == output_len:
                    self.samples.append((inp / normalize_scale, tgt / normalize_scale))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        inp, tgt = self.samples[idx]
        return torch.from_numpy(inp).float(), torch.from_numpy(tgt).float()


# ─────────────────────────────────────────────────────────────────────
# Inference helpers (used by tracking/tracker.py at runtime)
# ─────────────────────────────────────────────────────────────────────

def load_predictor(
    weights_path: str | Path,
    device: str = "cpu",
) -> Optional["LSTMMotionPredictor"]:
    """
    Load a trained LSTMMotionPredictor from a .pt file. Returns None if torch
    isn't installed or the file doesn't exist - the tracker should then fall
    back to linear extrapolation transparently.
    """
    if not _TORCH_AVAILABLE:
        logger.warning("[LSTM] torch not available; LSTM motion model disabled")
        return None
    path = Path(weights_path)
    if not path.is_file():
        logger.info(f"[LSTM] No weights at {path}; using linear extrapolation")
        return None
    try:
        ckpt = torch.load(str(path), map_location=device, weights_only=True)
        cfg = ckpt.get("config", {})
        model = LSTMMotionPredictor(
            input_len=cfg.get("input_len", DEFAULT_INPUT_LEN),
            output_len=cfg.get("output_len", DEFAULT_OUTPUT_LEN),
            hidden_dim=cfg.get("hidden_dim", DEFAULT_HIDDEN_DIM),
            num_layers=cfg.get("num_layers", DEFAULT_NUM_LAYERS),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        logger.info(
            f"[LSTM] Loaded motion model from {path.name} "
            f"(input_len={model.input_len}, output_len={model.output_len}, "
            f"hidden={model.hidden_dim}, device={device})"
        )
        return model
    except Exception as e:
        logger.warning(f"[LSTM] Failed to load {path}: {e}; using linear fallback")
        return None


def predict_future_positions(
    model: "LSTMMotionPredictor",
    trajectory_history: list[tuple],
    horizon: int,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """
    Given a track's recent (cx, cy, ts) history, predict the next `horizon`
    center positions using the LSTM. Returns absolute positions, not deltas.

    Returns None if not enough history (caller falls back to linear).
    """
    if model is None:
        return None
    needed_positions = model.input_len + 1
    if len(trajectory_history) < needed_positions:
        return None

    recent = np.array(
        [(cx, cy) for cx, cy, _ts in trajectory_history[-needed_positions:]],
        dtype=np.float32,
    )
    deltas = np.diff(recent, axis=0)               # (input_len, 2)
    x = torch.from_numpy(deltas).float().unsqueeze(0).to(device)  # (1, T, 2)

    with torch.no_grad():
        pred_deltas = model(x)[0].cpu().numpy()    # (output_len, 2)

    # Cumulative sum of deltas added to the last known position gives future positions.
    last_pos = recent[-1]
    future = last_pos + np.cumsum(pred_deltas, axis=0)
    return future[:horizon]
