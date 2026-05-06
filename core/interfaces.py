"""
Phantom Tracker — Shared Interfaces
=====================================
ALL 4 TEAM MEMBERS DEPEND ON THESE CONTRACTS.
Rules:
  - Any change MUST be agreed by all members
  - Never rename a field without updating all consumers
  - Add new fields freely, never remove existing ones
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# ============================================================
# DETECTION OUTPUT  (Divyansh → Dhruvish)
# ============================================================

@dataclass
class Detection:
    """Single detection from YOLO or Grounding DINO."""
    bbox: np.ndarray          # [x1, y1, x2, y2] pixel coords
    confidence: float         # 0.0–1.0
    class_name: str           # "person", "backpack", or open-vocab label
    class_id: int             # integer class ID (-1 for open-vocab)
    embedding: Optional[np.ndarray] = None

    @property
    def center(self) -> tuple:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def area(self) -> float:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])


@dataclass
class FrameDetections:
    """All detections for one frame."""
    frame_id: int
    timestamp: float
    detections: list[Detection]
    source: str = "yolo"           # "yolo" | "grounding_dino"
    inference_time_ms: float = 0.0
    camera_id: str = "default"     # which camera produced these detections (multi-cam)


# ============================================================
# TRACK STATE  (Dhruvish → Agastya, Dhruvish ↔ Dharmik)
# ============================================================

class TrackState(Enum):
    ACTIVE = "active"
    OCCLUDED = "occluded"
    LOST = "lost"
    DELETED = "deleted"


@dataclass
class Track:
    """Single tracked object with full state."""
    track_id: int
    state: TrackState
    bbox: np.ndarray                     # current or predicted [x1,y1,x2,y2]
    confidence: float
    class_name: str
    color: tuple                         # (R, G, B) persistent

    # Multi-camera identity
    camera_id: str = "default"           # which camera this track lives on
    global_id: Optional[int] = None      # cross-camera canonical id (assigned by multi-cam Re-ID)

    # Motion
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    predicted_position: Optional[np.ndarray] = None
    predicted_trajectory: list = field(default_factory=list)

    # History
    trajectory_history: list = field(default_factory=list)   # [(cx, cy, ts), ...]
    frames_since_seen: int = 0
    total_frames_tracked: int = 0
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0

    # Appearance (for Re-ID)
    appearance_gallery: list = field(default_factory=list)   # [(embedding, ts), ...]
    last_embedding: Optional[np.ndarray] = None

    # Speed
    instantaneous_speed: float = 0.0
    average_speed: float = 0.0

    @property
    def center(self) -> tuple:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def is_visible(self) -> bool:
        return self.state == TrackState.ACTIVE


# ============================================================
# RE-ID RESULT  (Dharmik → Dhruvish)
# ============================================================

@dataclass
class ReIDMatch:
    """One re-identification match."""
    new_detection_idx: int
    matched_track_id: int
    similarity_score: float
    is_confident: bool


@dataclass
class ReIDResult:
    """All Re-ID results for one frame."""
    frame_id: int
    matches: list[ReIDMatch]
    unmatched_detection_indices: list[int]
    inference_time_ms: float = 0.0


# ============================================================
# FRAME STATE  (Central pipeline object)
# ============================================================

@dataclass
class FrameState:
    """
    Complete state for one frame flowing through the pipeline:
    raw_frame → Detector → Tracker → ReID → Visualizer → output_frame
    """
    frame_id: int
    timestamp: float
    raw_frame: np.ndarray
    camera_id: str = "default"   # which camera this frame is from (multi-cam)

    detections: Optional[FrameDetections] = None
    active_tracks: list = field(default_factory=list)
    occluded_tracks: list = field(default_factory=list)
    lost_tracks: list = field(default_factory=list)
    reid_results: Optional[ReIDResult] = None
    output_frame: Optional[np.ndarray] = None

    @property
    def all_tracks(self) -> list:
        return self.active_tracks + self.occluded_tracks + self.lost_tracks


# ============================================================
# ANALYTICS  (Dhruvish → Agastya)
# ============================================================

@dataclass
class AnalyticsSnapshot:
    """Cumulative analytics for the dashboard."""
    frame_id: int
    timestamp: float
    camera_id: str = "default"
    track_speeds: dict = field(default_factory=dict)
    track_dwell_times: dict = field(default_factory=dict)
    heatmap_accumulator: Optional[np.ndarray] = None
    total_entries: int = 0
    total_exits: int = 0
    current_object_count: int = 0
    reid_events: list = field(default_factory=list)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class PipelineConfig:
    """Runtime configuration."""
    # Detection
    # Default points at our CrowdHuman-fine-tuned yolo11l checkpoint (76MB, lives at
    # weights/yolo11l_crowdhuman/best.pt). Trained 23 epochs on CrowdHuman train,
    # final mAP@50 = 0.876, recall = 0.786 on CrowdHuman val. Substantially better
    # person detection than off-the-shelf yolo11l/s on real-world video.
    #
    # Falls back automatically to "yolo11l.pt" (auto-downloads from Ultralytics)
    # if the local fine-tuned weights aren't present - kept this as a safety net
    # so the pipeline still works on machines without our weights.
    yolo_model: str = "weights/yolo11l_crowdhuman/best.pt"
    yolo_confidence: float = 0.5
    yolo_device: str = "auto"              # "auto" | "cpu" | "0" (GPU index)
    yolo_half: bool = False                # FP16 inference (GPU only, ~2x speedup)
    detect_classes: list = field(default_factory=list)  # e.g. ["person"] — empty = all
    use_grounding_dino: bool = False
    grounding_dino_prompt: str = "person."   # period-separated phrases, e.g. "person. backpack."
    grounding_dino_checkpoint: str = "IDEA-Research/grounding-dino-tiny"

    # Tracking
    tracker_type: str = "botsort"
    max_occlusion_frames: int = 30
    max_lost_frames: int = 300
    association_iou_threshold: float = 0.3

    # Tracking — boxmot specific
    boxmot_reid_weights: str = "osnet_x1_0_msmt17.pt"
    tracker_device: str = "cuda"
    tracker_half_precision: bool = False
    prediction_horizon: int = 15

    # LSTM motion predictor (replaces linear extrapolation when weights present).
    # Empty string -> always use linear fallback. See tracking/train_lstm.py to
    # produce the weights file from extracted MOT17 / demo trajectories.
    lstm_motion_weights: str = "weights/lstm_motion.pt"

    # Tracking — BoT-SORT tuning (indoor walking defaults)
    track_high_thresh: float = 0.45       # confidence for primary association (lower = keep more)
    track_low_thresh: float = 0.1         # confidence for secondary association pass
    new_track_thresh: float = 0.5         # min confidence to initialize a new track
    track_buffer: int = 90                # frames to keep lost track in boxmot (3s @ 30fps)
    match_thresh: float = 0.85            # IoU matching threshold (higher = stricter)
    proximity_thresh: float = 0.55        # spatial gate for ReID matching
    appearance_thresh: float = 0.2        # appearance gate for ReID (lower = more lenient)
    botsort_frame_rate: int = 30          # assumed frame rate for motion model

    # Re-ID
    reid_model: str = "osnet_x1_0"
    reid_confidence_threshold: float = 0.6
    gallery_max_size: int = 50
    temporal_decay_rate: float = 0.95

    # Cross-camera Re-ID (multi-camera mode)
    # Threshold is intentionally lower than single-camera reid_confidence_threshold:
    # cross-camera always involves viewpoint/lighting/scale differences that single-
    # camera matching does not, so OSNet similarity scores naturally land 0.1-0.2
    # lower for the same person across cameras vs the same person re-entering one camera.
    # 0.38 was chosen empirically: live phone+laptop testing showed legitimate
    # same-person matches landing as low as 0.400, with all confirmed matches in
    # the [0.44, 0.58] range. 0.38 leaves a 0.02 cushion below the lowest observed
    # legitimate match without dropping deep enough to admit obvious false positives.
    cross_camera_threshold: float = 0.38
    # min_transit_s = 0 lets a person be matched immediately on a new camera even if
    # the source camera still has them as an active track (brief FOV overlap, or the
    # source tracker lagging behind). The same-camera filter already prevents self-
    # handoff, so this only relaxes when the person is genuinely on a *different* cam.
    cross_camera_min_transit_s: float = 0.0
    cross_camera_max_transit_s: float = 30.0   # identity expires after this gap
    cross_camera_gallery_size: int = 20        # max embeddings per global identity

    # Visualization
    trail_length: int = 60
    show_ghost_outlines: bool = True
    show_predicted_path: bool = True
    show_trails: bool = True
    show_ids: bool = True
    show_fps: bool = True
    ghost_opacity: float = 0.4

    # Analytics
    heatmap_resolution: tuple = (64, 48)

    # Input
    input_source: int | str = 0
    frame_width: int = 1280
    frame_height: int = 720
