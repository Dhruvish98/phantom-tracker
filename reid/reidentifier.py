from __future__ import annotations
"""
reid/reidentifier.py — Owner: DHARMIK
=======================================
Re-Identification module for Phantom Tracker.

Pipeline:
  1. extract_embedding()   — crop person from frame → 512-d OSNet embedding
  2. update_gallery()      — maintain per-track appearance gallery w/ temporal decay
  3. match()               — cosine similarity matching of new detections vs lost tracks
  4. apply_reid_results()  — called by tracker to reactivate matched lost tracks

Integrates with:
  - Dhruvish's tracker.py  → receives lost_tracks (List[Track]), sends back ReIDResult
  - core/interfaces.py     → uses Track, TrackState, ReIDMatch, ReIDResult, PipelineConfig

MVP target: Person leaves frame, returns within 5-10 sec → same ID restored.
"""

import time
import numpy as np
from typing import Optional
from core.interfaces import (
    Track, TrackState, FrameDetections, ReIDMatch, ReIDResult, PipelineConfig
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# OSNet Model Loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_osnet(model_name: str, device: str):
    """
    Load OSNet via torchreid's FeatureExtractor.
    Handles both PyPI version (torchreid) and GitHub version (deep-person-reid).
    Falls back gracefully if torchreid is not installed.

    Model options (fastest → most accurate):
      osnet_x0_25   ~0.3M params  — fastest, lower accuracy
      osnet_x0_5    ~1.1M params
      osnet_x0_75   ~1.7M params
      osnet_x1_0    ~2.2M params  — recommended (best speed/accuracy tradeoff)
      osnet_ibn_x1_0               — better cross-domain generalisation
    """
    FeatureExtractor = None

    # Try PyPI version first (pip install torchreid)
    try:
        from torchreid.reid.utils import FeatureExtractor
        logger.info("[ReID] Using torchreid PyPI version")
    except ImportError:
        pass

    # Try GitHub version (pip install git+https://github.com/KaiyangZhou/...)
    if FeatureExtractor is None:
        try:
            from torchreid.utils import FeatureExtractor
            logger.info("[ReID] Using torchreid GitHub version")
        except ImportError:
            pass

    if FeatureExtractor is None:
        logger.warning(
            "[ReID] torchreid not installed — Re-ID disabled.\n"
            "       Install with:  pip install torchreid"
        )
        return None

    try:
        extractor = FeatureExtractor(
            model_name=model_name,
            model_path='',          # '' = download pretrained weights automatically
            device=device
        )
        logger.info(f"[ReID] OSNet loaded: {model_name} on {device}")
        return extractor
    except Exception as e:
        logger.warning(f"[ReID] Model load failed: {e}. Re-ID disabled.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Feature Bank
# ──────────────────────────────────────────────────────────────────────────────

class FeatureBank:
    """
    Per-track gallery of appearance embeddings with temporal decay.

    Each track maintains a rolling gallery of (embedding, timestamp) pairs.
    When computing similarity, recent embeddings are weighted more heavily
    via an exponential decay schedule.

    Design decisions:
      - Max gallery size (default 50): prevents unbounded memory growth
      - Temporal decay rate (default 0.95): ~14 entries back = half weight
      - We keep the gallery as a list (oldest → newest) for simple decay indexing
    """

    def __init__(self, max_size: int = 50, decay_rate: float = 0.95):
        self.max_size   = max_size
        self.decay_rate = decay_rate

    def add(self, track: Track, embedding: np.ndarray):
        """Add a new embedding to the track's gallery, pruning if over capacity."""
        track.appearance_gallery.append((embedding, time.time()))
        track.last_embedding = embedding
        if len(track.appearance_gallery) > self.max_size:
            track.appearance_gallery.pop(0)   # remove oldest

    def similarity(self, query_emb: np.ndarray, track: Track) -> float:
        """
        Compute weighted cosine similarity between query and a track's gallery.

        Temporal weighting: entry at position i (0=oldest) gets weight:
            w_i = decay_rate ^ (N - 1 - i)
        so the most recent entry always gets weight 1.0, and older entries
        decay exponentially.

        Returns value in [-1, 1]. Typical match threshold: 0.6.
        """
        gallery = track.appearance_gallery
        if not gallery:
            return -1.0

        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        N = len(gallery)

        weighted_sim = 0.0
        weight_sum   = 0.0

        for i, (emb, _ts) in enumerate(gallery):
            e    = emb / (np.linalg.norm(emb) + 1e-8)
            sim  = float(np.dot(q, e))
            w    = self.decay_rate ** (N - 1 - i)   # recent = higher weight
            weighted_sim += w * sim
            weight_sum   += w

        return weighted_sim / (weight_sum + 1e-8)

    def best_match(
        self,
        query_emb: np.ndarray,
        lost_tracks: list,
        exclude_ids: set,
        threshold: float
    ) -> tuple[int, float]:
        """
        Find the best matching lost track for a query embedding.

        Args:
            query_emb    : query embedding (512-d, already L2-normalised)
            lost_tracks  : list of Track objects in LOST state
            exclude_ids  : track IDs already matched this frame (skip them)
            threshold    : minimum similarity to accept a match

        Returns:
            (matched_track_id, similarity_score)
            matched_track_id == -1 if no match found above threshold
        """
        best_id    = -1
        best_score = -1.0

        for track in lost_tracks:
            if track.track_id in exclude_ids:
                continue
            if not track.appearance_gallery:
                continue

            score = self.similarity(query_emb, track)
            if score > best_score:
                best_score = score
                best_id    = track.track_id

        if best_id >= 0 and best_score >= threshold:
            return best_id, best_score
        return -1, best_score


# ──────────────────────────────────────────────────────────────────────────────
# ReIdentifier — Main Class (used by main.py and tracker.py)
# ──────────────────────────────────────────────────────────────────────────────

class ReIdentifier:
    """
    Dharmik's Re-ID module. Drop-in replacement for the stub in the repo.

    Called by main.py (PhantomTracker.process_frame):
        state.reid_results = self.reidentifier.match(
            state.detections, state.lost_tracks,
            {t.track_id for t in state.active_tracks}
        )

    Also called by tracker.py for gallery updates:
        self.reidentifier.update_gallery(track, frame)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Detect device
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[ReID] Device: {self.device}")

        # Load OSNet
        self.model = _load_osnet(config.reid_model, self.device)

        # Feature bank shared across all tracks
        self.bank = FeatureBank(
            max_size=config.gallery_max_size,
            decay_rate=config.temporal_decay_rate
        )

        # Stats for logging / ablation
        self._total_queries    = 0
        self._total_matches    = 0
        self._total_rejections = 0

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API — called from main.py and tracker.py
    # ──────────────────────────────────────────────────────────────────────

    def extract_embedding(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-d L2-normalised OSNet embedding from a person crop.

        Args:
            frame : full BGR frame (H, W, 3) from OpenCV
            bbox  : np.ndarray [x1, y1, x2, y2]

        Returns:
            np.ndarray shape (512,) float32, or None if model unavailable
        """
        if self.model is None:
            return None

        crop = self._safe_crop(frame, bbox)
        if crop is None:
            return None

        try:
            import cv2
            # torchreid FeatureExtractor accepts list of BGR numpy arrays
            crop_resized = cv2.resize(crop, (128, 256))
            features = self.model([crop_resized])      # returns torch.Tensor (1, 512)
            emb = features[0].cpu().numpy()
            # L2 normalise (OSNet outputs should already be normalised, but enforce)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return emb.astype(np.float32)
        except Exception as e:
            logger.debug(f"[ReID] Embedding extraction failed: {e}")
            return None

    def update_gallery(self, track: Track, frame: np.ndarray):
        """
        Extract embedding for an ACTIVE track and add to its gallery.

        Called by tracker after each successful detection association.
        Should be called every N frames (every frame is fine for MVP).

        Args:
            track : Track object (must be ACTIVE state for meaningful crop)
            frame : current BGR frame
        """
        emb = self.extract_embedding(frame, track.bbox)
        if emb is not None:
            self.bank.add(track, emb)

    def match(
        self,
        detections: FrameDetections,
        lost_tracks: list,
        active_track_ids: set,
        frame: Optional[np.ndarray] = None
    ) -> ReIDResult:
        """
        Core Re-ID matching step. Compare each unmatched detection against
        all lost track galleries and return confident matches.

        Args:
            detections       : FrameDetections from Divyansh's detector
            lost_tracks      : List[Track] in LOST state (from Dhruvish's tracker)
            active_track_ids : set of track_ids already matched this frame
            frame            : raw BGR frame (needed to extract embeddings on-the-fly
                               if detections don't carry embeddings)

        Returns:
            ReIDResult with matches and unmatched detection indices.
            Passed back to tracker.apply_reid_results().
        """
        t_start = time.time()
        matches:   list[ReIDMatch] = []
        unmatched: list[int]       = []

        # Nothing to match against
        if not lost_tracks:
            return ReIDResult(
                frame_id=detections.frame_id,
                matches=[],
                unmatched_detection_indices=list(range(len(detections.detections))),
                inference_time_ms=0.0
            )

        # Track IDs already re-identified this frame (prevent double assignment)
        claimed_ids: set[int] = set(active_track_ids)

        for det_idx, det in enumerate(detections.detections):
            self._total_queries += 1

            # --- Step 1: Get embedding for this detection ---
            query_emb = det.embedding   # may be None if detector didn't compute it

            if query_emb is None and frame is not None:
                # Extract on-the-fly from the raw frame
                query_emb = self.extract_embedding(frame, det.bbox)

            if query_emb is None:
                unmatched.append(det_idx)
                continue

            # --- Step 2: Find best matching lost track ---
            best_id, best_score = self.bank.best_match(
                query_emb=query_emb,
                lost_tracks=lost_tracks,
                exclude_ids=claimed_ids,
                threshold=self.config.reid_confidence_threshold
            )

            # --- Step 3: Record result ---
            if best_id >= 0:
                matches.append(ReIDMatch(
                    new_detection_idx=det_idx,
                    matched_track_id=best_id,
                    similarity_score=best_score,
                    is_confident=True
                ))
                claimed_ids.add(best_id)   # prevent this track being matched again
                self._total_matches += 1
                logger.info(
                    f"[ReID] ✓ detection[{det_idx}] → Track #{best_id}  "
                    f"sim={best_score:.3f}  frame={detections.frame_id}"
                )
            else:
                unmatched.append(det_idx)
                self._total_rejections += 1
                logger.debug(
                    f"[ReID] ✗ detection[{det_idx}] best_sim={best_score:.3f} "
                    f"(threshold={self.config.reid_confidence_threshold})"
                )

        elapsed_ms = (time.time() - t_start) * 1000

        return ReIDResult(
            frame_id=detections.frame_id,
            matches=matches,
            unmatched_detection_indices=unmatched,
            inference_time_ms=elapsed_ms
        )

    def get_stats(self) -> dict:
        """Return running Re-ID statistics. Useful for ablation study."""
        match_rate = (self._total_matches / self._total_queries
                      if self._total_queries > 0 else 0.0)
        return {
            "total_queries":    self._total_queries,
            "total_matches":    self._total_matches,
            "total_rejections": self._total_rejections,
            "match_rate":       match_rate,
        }

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_crop(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Safely crop a person from the frame, clamping to frame boundaries.
        Returns None if the crop area is degenerate (too small / out of bounds).
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        x1 = int(max(0, bbox[0]))
        y1 = int(max(0, bbox[1]))
        x2 = int(min(w, bbox[2]))
        y2 = int(min(h, bbox[3]))

        if x2 - x1 < 16 or y2 - y1 < 16:   # crop too small to be meaningful
            return None

        return frame[y1:y2, x1:x2]
