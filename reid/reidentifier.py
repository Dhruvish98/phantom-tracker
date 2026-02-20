"""
Re-Identification Module — Owner: DHARMIK
============================================
Matches new detections against lost tracks using appearance embeddings.

Pipeline: extract embedding → compare to lost track gallery → return matches

MVP target: Basic Re-ID working for person leaving and returning within 10 sec.

TODO (Dharmik):
  [ ] Get OSNet / FastReID running and extracting embeddings
  [ ] Build feature bank with temporal decay
  [ ] Tune confidence threshold on real webcam footage
  [ ] Implement GAN augmentation for training robustness (post-MVP)
  [ ] Run ablation study: tracker with vs without Re-ID (post-MVP)
"""

import numpy as np
import time
from typing import Optional
from core.interfaces import (
    Detection, FrameDetections, Track, TrackState,
    ReIDMatch, ReIDResult, PipelineConfig
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ReIdentifier:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.transform = None
        self._load_model()

    def _load_model(self):
        """
        Load Re-ID model (OSNet or FastReID).

        DHARMIK — TWO APPROACHES:

        Approach A: torchreid (simpler)
          pip install torchreid
          from torchreid.utils import FeatureExtractor
          self.model = FeatureExtractor(
              model_name='osnet_x1_0',
              model_path='path/to/weights.pth',
              device='cuda')

        Approach B: FastReID (more powerful, more complex)
          pip install fastreid
          Gives you more model options and better accuracy

        For MVP: Use torchreid — it's a 3-line setup.
        """
        try:
            from torchreid.utils import FeatureExtractor
            self.model = FeatureExtractor(
                model_name=self.config.reid_model,
                device='cuda'  # falls back to cpu if no GPU
            )
            logger.info(f"Re-ID model loaded: {self.config.reid_model}")
        except ImportError:
            logger.warning(
                "torchreid not installed. Re-ID disabled. "
                "Install: pip install torchreid"
            )
        except Exception as e:
            logger.warning(f"Re-ID model load failed: {e}. Re-ID disabled.")

    def extract_embedding(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract appearance embedding for a detected object.

        Args:
            frame: full BGR frame
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            embedding vector (typically 512-dim) or None if model unavailable
        """
        if self.model is None:
            return None

        # Crop the detected object from the frame
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        try:
            # torchreid expects a list of image paths or numpy arrays
            import cv2
            # Resize to model input size (256x128 is standard for person ReID)
            crop_resized = cv2.resize(crop, (128, 256))
            features = self.model([crop_resized])
            return features[0].cpu().numpy()
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
            return None

    def match(self, detections: FrameDetections, lost_tracks: list[Track],
              active_track_ids: set[int]) -> ReIDResult:
        """
        Core Re-ID matching. Called when there are unmatched detections
        AND lost tracks that might be the same objects.

        Args:
            detections: current frame detections
            lost_tracks: tracks in LOST state
            active_track_ids: IDs already matched this frame (skip these)

        Returns:
            ReIDResult with matches and unmatched indices
        """
        start = time.time()
        matches = []
        unmatched = []

        if self.model is None:
            # No model available — return empty results
            return ReIDResult(
                frame_id=detections.frame_id,
                matches=[],
                unmatched_detection_indices=list(range(len(detections.detections))),
                inference_time_ms=0.0
            )

        # Extract embeddings for all current detections
        det_embeddings = []
        for det in detections.detections:
            # TODO (Dharmik): Get the raw frame from FrameState for cropping
            # For now, embeddings are extracted if available in the detection
            emb = det.embedding
            det_embeddings.append(emb)

        # Compare each unmatched detection against lost track galleries
        for det_idx, det in enumerate(detections.detections):
            det_emb = det_embeddings[det_idx]
            if det_emb is None:
                unmatched.append(det_idx)
                continue

            best_match_id = -1
            best_similarity = -1.0

            for track in lost_tracks:
                if track.track_id in active_track_ids:
                    continue
                if not track.appearance_gallery:
                    continue

                # Compare against gallery with temporal weighting
                similarity = self._compute_gallery_similarity(
                    det_emb, track.appearance_gallery)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = track.track_id

            if best_match_id >= 0 and best_similarity >= self.config.reid_confidence_threshold:
                matches.append(ReIDMatch(
                    new_detection_idx=det_idx,
                    matched_track_id=best_match_id,
                    similarity_score=best_similarity,
                    is_confident=True
                ))
                logger.info(
                    f"Re-ID match: detection {det_idx} → Track #{best_match_id} "
                    f"(similarity: {best_similarity:.3f})")
            else:
                unmatched.append(det_idx)

        elapsed = (time.time() - start) * 1000

        return ReIDResult(
            frame_id=detections.frame_id,
            matches=matches,
            unmatched_detection_indices=unmatched,
            inference_time_ms=elapsed
        )

    def _compute_gallery_similarity(self, query_emb: np.ndarray,
                                     gallery: list[tuple]) -> float:
        """
        Compare query embedding against a track's appearance gallery.

        Gallery entries are (embedding, timestamp). Recent entries get higher weight.

        DHARMIK: This is where temporal decay happens.
        """
        if not gallery:
            return -1.0

        similarities = []
        weights = []

        for i, (emb, ts) in enumerate(gallery):
            # Cosine similarity
            sim = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
            similarities.append(sim)

            # Temporal weight: more recent = higher weight
            recency = len(gallery) - i  # newer entries have higher recency
            weight = self.config.temporal_decay_rate ** (len(gallery) - recency)
            weights.append(weight)

        similarities = np.array(similarities)
        weights = np.array(weights)
        weights = weights / weights.sum()  # normalize

        return float(np.dot(similarities, weights))

    def update_gallery(self, track: Track, frame: np.ndarray):
        """
        Add current appearance to a track's gallery.
        Call this for every active track each frame.

        DHARMIK: This should be called from the tracker after association.
        """
        emb = self.extract_embedding(frame, track.bbox)
        if emb is not None:
            track.appearance_gallery.append((emb, time.time()))
            track.last_embedding = emb
            # Prune old entries
            if len(track.appearance_gallery) > self.config.gallery_max_size:
                track.appearance_gallery.pop(0)
