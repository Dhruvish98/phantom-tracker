from __future__ import annotations
"""
Phantom Tracker — Shared Interfaces
=====================================
ALL 4 TEAM MEMBERS DEPEND ON THESE CONTRACTS.
Rules:
  - Any change MUST be agreed by all members
  - Never rename a field without updating all consumers
  - Add new fields freely, never remove existing ones
"""

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
    yolo_model: str = "yolo11n.pt"
    yolo_confidence: float = 0.5
    use_grounding_dino: bool = False

    # Tracking
    tracker_type: str = "botsort"
    max_occlusion_frames: int = 30
    max_lost_frames: int = 300
    association_iou_threshold: float = 0.3

    # Re-ID
    reid_model: str = "osnet_x1_0"
    reid_confidence_threshold: float = 0.6
    gallery_max_size: int = 50
    temporal_decay_rate: float = 0.95

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
