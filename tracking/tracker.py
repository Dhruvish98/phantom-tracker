"""
Tracking Module — Owner: DHRUVISH
====================================
Multi-object tracker with occlusion state machine and motion prediction.

Pipeline: detections → boxmot BoT-SORT (association + Kalman) → state machine → predict next

Implementation:
  [x] BoT-SORT via boxmot with deep appearance features (ReID)
  [x] Occlusion state machine (Active → Occluded → Lost → Deleted)
  [x] EMA-smoothed velocity and trajectory prediction
  [x] Ghost position output for Agastya's visualization
  [ ] LSTM motion model for trajectory prediction (post-MVP)
"""

import numpy as np
import time
from pathlib import Path
from core.interfaces import (
    Track, TrackState, FrameDetections, ReIDResult,
    AnalyticsSnapshot, PipelineConfig
)
from utils.logger import setup_logger
from utils.colors import generate_unique_color

logger = setup_logger(__name__)

# Velocity smoothing factor (0 = full history, 1 = only current frame)
_VELOCITY_ALPHA = 0.3
# Assumed FPS for per-frame time step when timestamps are unavailable
_DEFAULT_FPS = 30.0


class Tracker:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1  # only used by IoU fallback
        self.deleted_tracks: dict[int, Track] = {}
        self._boxmot_tracker = None
        self._use_fallback = False
        self._init_tracker()

        # Analytics accumulators
        self.total_entries = 0
        self.total_exits = 0
        self.heatmap = np.zeros(config.heatmap_resolution[::-1], dtype=np.float32)

    def _init_tracker(self):
        """Initialize BoT-SORT via boxmot. Falls back to IoU stub if unavailable."""
        try:
            from boxmot import BotSort
            self._boxmot_tracker = BotSort(
                reid_weights=Path(self.config.boxmot_reid_weights),
                device=self.config.tracker_device,
                half=self.config.tracker_half_precision,
            )
            logger.info(f"BoT-SORT initialized (device={self.config.tracker_device}, "
                        f"reid_weights={self.config.boxmot_reid_weights})")
        except ImportError:
            logger.warning("boxmot not installed. Falling back to IoU-based tracking. "
                           "Install: pip install boxmot")
            self._use_fallback = True
        except Exception as e:
            logger.warning(f"BoT-SORT init failed: {e}. Falling back to IoU-based tracking.")
            self._use_fallback = True

    # ================================================================
    # PUBLIC API
    # ================================================================

    def update(self, frame_detections: FrameDetections,
               frame: np.ndarray = None) -> tuple[list, list, list]:
        """
        Core update step. Called every frame by main.py.

        Args:
            frame_detections: output from Divyansh's Detector
            frame: raw BGR frame (needed by boxmot for ReID embeddings)

        Returns:
            (active_tracks, occluded_tracks, lost_tracks)
        """
        if self._use_fallback:
            return self._update_fallback(frame_detections)
        return self._update_boxmot(frame_detections, frame)

    def apply_reid_results(self, reid_results: ReIDResult):
        """
        Apply Dharmik's Re-ID matches: restore Lost tracks to Active.

        When Re-ID says "new detection #3 matches lost track #7",
        we reactivate track #7 with the new detection's bbox.
        """
        for match in reid_results.matches:
            if not match.is_confident:
                continue
            if match.matched_track_id in self.tracks:
                track = self.tracks[match.matched_track_id]
                track.state = TrackState.ACTIVE
                track.frames_since_seen = 0
                logger.info(f"Re-ID: Track #{track.track_id} reactivated "
                            f"(confidence: {match.similarity_score:.1%})")

    def get_analytics(self, frame_id: int, timestamp: float) -> AnalyticsSnapshot:
        """Build analytics snapshot for Agastya's dashboard."""
        speeds = {tid: t.instantaneous_speed for tid, t in self.tracks.items()
                  if t.state == TrackState.ACTIVE}
        dwell = {tid: timestamp - t.first_seen_timestamp for tid, t in self.tracks.items()
                 if t.state != TrackState.DELETED}

        return AnalyticsSnapshot(
            frame_id=frame_id, timestamp=timestamp,
            track_speeds=speeds, track_dwell_times=dwell,
            heatmap_accumulator=self.heatmap.copy(),
            total_entries=self.total_entries, total_exits=self.total_exits,
            current_object_count=sum(1 for t in self.tracks.values()
                                     if t.state in (TrackState.ACTIVE, TrackState.OCCLUDED)),
        )

    # ================================================================
    # BOXMOT BoT-SORT TRACKING
    # ================================================================

    def _update_boxmot(self, frame_detections: FrameDetections,
                       frame: np.ndarray) -> tuple[list, list, list]:
        """
        Core tracking update using boxmot BoT-SORT.

        Flow:
          1. Convert FrameDetections → numpy (N, 6) for boxmot
          2. Call boxmot tracker.update(dets, frame)
          3. Parse output → update/create Track objects
          4. Detect disappeared tracks → run state machine
          5. Compute velocity, trajectory, predicted positions
          6. Classify into active/occluded/lost
        """
        timestamp = frame_detections.timestamp

        # Step 1: Convert detections
        dets_array = self._detections_to_numpy(frame_detections)

        # Step 2: Run boxmot
        if frame is None:
            logger.warning("No frame provided to boxmot; skipping tracking this frame")
            self._age_unmatched_tracks(timestamp, matched_ids=set())
            return self._classify_tracks()

        results = self._boxmot_tracker.update(dets_array, frame)
        # results shape: (M, 8) — [x1, y1, x2, y2, track_id, conf, cls_id, det_index]

        # Step 3: Process boxmot output
        current_ids = set()

        if results is not None and len(results) > 0:
            for row in results:
                x1, y1, x2, y2 = row[0:4]
                track_id = int(row[4])
                conf = float(row[5])
                cls_id = int(row[6])
                bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                class_name = self._resolve_class_name(frame_detections, cls_id, row)

                current_ids.add(track_id)

                if track_id in self.tracks:
                    self._update_existing_track(track_id, bbox, conf, class_name, timestamp)
                else:
                    self._create_new_track(track_id, bbox, conf, class_name, timestamp)

        # Step 4: Age disappeared tracks through state machine
        self._age_unmatched_tracks(timestamp, matched_ids=current_ids)

        # Step 5: Update predicted trajectories for active tracks
        self._update_predictions()

        # Step 6: Classify and return
        return self._classify_tracks()

    # ================================================================
    # TRACK LIFECYCLE
    # ================================================================

    def _create_new_track(self, track_id: int, bbox: np.ndarray,
                          conf: float, class_name: str, timestamp: float):
        """Create a new Track object for a track ID first seen by boxmot."""
        color = generate_unique_color(track_id)
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        track = Track(
            track_id=track_id,
            state=TrackState.ACTIVE,
            bbox=bbox,
            confidence=conf,
            class_name=class_name,
            color=color,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            trajectory_history=[(cx, cy, timestamp)],
            total_frames_tracked=1,
        )
        self.tracks[track_id] = track
        self.total_entries += 1
        logger.debug(f"New track #{track_id}: {class_name} @ ({cx:.0f}, {cy:.0f})")

    def _update_existing_track(self, track_id: int, bbox: np.ndarray,
                               conf: float, class_name: str, timestamp: float):
        """Update an existing Track with new boxmot detection."""
        track = self.tracks[track_id]
        old_cx, old_cy = track.center

        # Core fields
        track.bbox = bbox
        track.confidence = conf
        track.state = TrackState.ACTIVE
        track.frames_since_seen = 0
        track.total_frames_tracked += 1

        # Time delta
        dt = timestamp - track.last_seen_timestamp
        if dt <= 0:
            dt = 1.0 / _DEFAULT_FPS
        track.last_seen_timestamp = timestamp

        # New center
        cx, cy = track.center

        # EMA-smoothed velocity (pixels / second)
        raw_vx = (cx - old_cx) / dt
        raw_vy = (cy - old_cy) / dt
        track.velocity = (
            _VELOCITY_ALPHA * np.array([raw_vx, raw_vy], dtype=np.float64)
            + (1 - _VELOCITY_ALPHA) * track.velocity
        )

        # Speed
        pixel_disp = np.sqrt((cx - old_cx) ** 2 + (cy - old_cy) ** 2)
        track.instantaneous_speed = pixel_disp / dt
        n = track.total_frames_tracked
        if n > 1:
            track.average_speed = (
                track.average_speed * (n - 1) + track.instantaneous_speed
            ) / n
        else:
            track.average_speed = track.instantaneous_speed

        # Trajectory history
        track.trajectory_history.append((cx, cy, timestamp))
        if len(track.trajectory_history) > self.config.trail_length:
            track.trajectory_history.pop(0)

        # Heatmap
        self._update_heatmap(cx, cy)

    # ================================================================
    # STATE MACHINE
    # ================================================================

    def _age_unmatched_tracks(self, timestamp: float, matched_ids: set[int]):
        """
        Transition unmatched tracks through the occlusion state machine.

        State machine:
          ACTIVE ---(3 frames unseen)---> OCCLUDED
          OCCLUDED ---(max_occlusion_frames)---> LOST
          LOST ---(max_lost_frames)---> DELETED (removed)
        """
        to_delete = []

        for tid, track in self.tracks.items():
            if tid in matched_ids:
                continue  # seen this frame — skip

            track.frames_since_seen += 1

            if track.state == TrackState.ACTIVE:
                if track.frames_since_seen > 3:
                    track.state = TrackState.OCCLUDED
                    logger.debug(f"Track #{tid}: ACTIVE -> OCCLUDED")
                self._extrapolate_position(track)

            elif track.state == TrackState.OCCLUDED:
                self._extrapolate_position(track)
                if track.frames_since_seen > self.config.max_occlusion_frames:
                    track.state = TrackState.LOST
                    logger.debug(f"Track #{tid}: OCCLUDED -> LOST "
                                 f"(unseen for {track.frames_since_seen} frames)")

            elif track.state == TrackState.LOST:
                if track.frames_since_seen > self.config.max_lost_frames:
                    to_delete.append(tid)
                    self.total_exits += 1
                    logger.debug(f"Track #{tid}: LOST -> DELETED")

        for tid in to_delete:
            self.deleted_tracks[tid] = self.tracks.pop(tid)
            self.deleted_tracks[tid].state = TrackState.DELETED

    # ================================================================
    # MOTION PREDICTION
    # ================================================================

    def _extrapolate_position(self, track: Track):
        """
        Extrapolate bbox for an unseen track using EMA velocity.
        Provides the ghost bbox for Agastya's visualization.
        """
        cx, cy = track.center
        vx, vy = track.velocity
        dt = 1.0 / _DEFAULT_FPS

        pred_cx = cx + vx * dt
        pred_cy = cy + vy * dt

        # Clamp to frame boundaries
        pred_cx = float(np.clip(pred_cx, 0, self.config.frame_width))
        pred_cy = float(np.clip(pred_cy, 0, self.config.frame_height))

        w = track.bbox[2] - track.bbox[0]
        h = track.bbox[3] - track.bbox[1]
        track.bbox = np.array([
            pred_cx - w / 2, pred_cy - h / 2,
            pred_cx + w / 2, pred_cy + h / 2
        ], dtype=np.float32)
        track.predicted_position = np.array([pred_cx, pred_cy])

    def _update_predictions(self):
        """Compute predicted_trajectory for active tracks (future path visualization)."""
        horizon = self.config.prediction_horizon
        dt = 1.0 / _DEFAULT_FPS

        for track in self.tracks.values():
            if track.state != TrackState.ACTIVE:
                continue

            cx, cy = track.center
            vx, vy = track.velocity

            trajectory = []
            for step in range(1, horizon + 1):
                future_cx = cx + vx * dt * step
                future_cy = cy + vy * dt * step
                trajectory.append((future_cx, future_cy))

            track.predicted_trajectory = trajectory

    # ================================================================
    # HELPERS
    # ================================================================

    @staticmethod
    def _detections_to_numpy(frame_detections: FrameDetections) -> np.ndarray:
        """Convert FrameDetections → (N, 6) numpy array for boxmot.
        Format: [x1, y1, x2, y2, confidence, class_id]
        """
        if not frame_detections.detections:
            return np.empty((0, 6), dtype=np.float32)

        dets = []
        for det in frame_detections.detections:
            dets.append([
                det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
                det.confidence,
                det.class_id
            ])
        return np.array(dets, dtype=np.float32)

    @staticmethod
    def _resolve_class_name(frame_detections: FrameDetections,
                            cls_id: int, row: np.ndarray) -> str:
        """Look up class_name from detection index or class_id."""
        if len(row) > 7:
            det_idx = int(row[7])
            if 0 <= det_idx < len(frame_detections.detections):
                return frame_detections.detections[det_idx].class_name

        for det in frame_detections.detections:
            if det.class_id == cls_id:
                return det.class_name

        return f"class_{cls_id}"

    def _update_heatmap(self, cx: float, cy: float):
        """Increment heatmap at the given center position."""
        hx = int(cx / self.config.frame_width * self.config.heatmap_resolution[0])
        hy = int(cy / self.config.frame_height * self.config.heatmap_resolution[1])
        hx = int(np.clip(hx, 0, self.config.heatmap_resolution[0] - 1))
        hy = int(np.clip(hy, 0, self.config.heatmap_resolution[1] - 1))
        self.heatmap[hy, hx] += 1

    def _classify_tracks(self) -> tuple[list, list, list]:
        """Partition self.tracks into (active, occluded, lost) lists."""
        active, occluded, lost = [], [], []
        for track in self.tracks.values():
            if track.state == TrackState.ACTIVE:
                active.append(track)
            elif track.state == TrackState.OCCLUDED:
                occluded.append(track)
            elif track.state == TrackState.LOST:
                lost.append(track)
        return active, occluded, lost

    # ================================================================
    # FALLBACK: IoU-based tracking (when boxmot is not available)
    # ================================================================

    def _update_fallback(self, frame_detections: FrameDetections) -> tuple[list, list, list]:
        """Simple IoU nearest-neighbor tracking. Used when boxmot is unavailable."""
        timestamp = frame_detections.timestamp
        matched_ids = set()

        for det in frame_detections.detections:
            track = self._create_or_update_track_iou(det, timestamp)
            matched_ids.add(track.track_id)

        self._age_unmatched_tracks(timestamp, matched_ids=matched_ids)
        self._update_predictions()

        return self._classify_tracks()

    def _create_or_update_track_iou(self, det, timestamp: float) -> Track:
        """IoU-based nearest-neighbor matching (fallback path)."""
        best_match = None
        best_iou = self.config.association_iou_threshold

        for tid, track in self.tracks.items():
            if track.state in (TrackState.ACTIVE, TrackState.OCCLUDED):
                iou = self._compute_iou(det.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = tid

        if best_match is not None:
            self._update_existing_track(
                best_match, det.bbox, det.confidence, det.class_name, timestamp)
            return self.tracks[best_match]
        else:
            track_id = self.next_track_id
            self.next_track_id += 1
            self._create_new_track(track_id, det.bbox, det.confidence,
                                   det.class_name, timestamp)
            return self.tracks[track_id]

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
