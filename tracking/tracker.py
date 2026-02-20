"""
Tracking Module — Owner: DHRUVISH
====================================
Multi-object tracker with occlusion state machine and motion prediction.

Pipeline: detections → association (Hungarian) → state update → predict next

MVP target: BoT-SORT running live, IDs persist through simple occlusions.

TODO (Dhruvish):
  [ ] Get BoT-SORT / ByteTrack baseline running on YOLO detections
  [ ] Implement occlusion state machine (Active → Occluded → Lost → Deleted)
  [ ] Tune Kalman filter parameters for indoor walking speed
  [ ] Add LSTM motion model for trajectory prediction (post-MVP)
  [ ] Implement ghost position output for Agastya's visualization
"""

import numpy as np
import time
from collections import defaultdict
from core.interfaces import (
    Track, TrackState, FrameDetections, ReIDResult,
    AnalyticsSnapshot, PipelineConfig
)
from utils.logger import setup_logger
from utils.colors import generate_unique_color

logger = setup_logger(__name__)


class Tracker:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tracks: dict[int, Track] = {}        # track_id → Track
        self.next_track_id = 1
        self.deleted_tracks: dict[int, Track] = {}  # for analytics history
        self._init_tracker()

        # Analytics accumulators
        self.total_entries = 0
        self.total_exits = 0
        self.heatmap = np.zeros(config.heatmap_resolution[::-1], dtype=np.float32)

    def _init_tracker(self):
        """
        Initialize the underlying tracker (BoT-SORT or ByteTrack).

        DHRUVISH — TWO APPROACHES (pick one):

        Approach A: Use ultralytics built-in tracker
          - Simpler, works out of the box
          - results = model.track(frame, tracker="botsort.yaml", persist=True)
          - But gives you less control over state management

        Approach B: Use standalone BoT-SORT / ByteTrack
          - pip install boxmot
          - More control, can plug in custom Re-ID and motion model
          - This is the recommended approach for the full project

        For MVP: Start with Approach A (get it working fast), then migrate
        to Approach B when you need custom occlusion handling.
        """
        self._use_builtin_tracker = True  # flip to False when using boxmot
        logger.info(f"Tracker initialized: {self.config.tracker_type}")

    def update(self, frame_detections: FrameDetections) -> tuple[list, list, list]:
        """
        Core update step. Called every frame.

        Args:
            frame_detections: output from Divyansh's Detector

        Returns:
            (active_tracks, occluded_tracks, lost_tracks)
        """
        if self._use_builtin_tracker:
            return self._update_builtin(frame_detections)
        else:
            return self._update_custom(frame_detections)

    # ----------------------------------------------------------------
    # APPROACH A: Ultralytics built-in tracker (for MVP)
    # ----------------------------------------------------------------

    def _update_builtin(self, frame_detections: FrameDetections) -> tuple[list, list, list]:
        """
        MVP tracking using ultralytics built-in track() method.

        DHRUVISH: This is your starting point. The ultralytics tracker
        handles association internally. You extract track IDs from the
        results and wrap them in our Track dataclass.

        Limitation: No occlusion state machine — you'll need to add that
        on top by tracking which IDs disappear and reappear.
        """
        # TODO (Dhruvish): Replace this stub with actual tracking logic
        #
        # Step 1: Convert frame_detections into the format your tracker expects
        # Step 2: Run tracker update
        # Step 3: Map tracker output to our Track objects
        # Step 4: Update state machine (Active/Occluded/Lost)
        # Step 5: Update trajectory history and speed calculations

        active = []
        occluded = []
        lost = []

        for det in frame_detections.detections:
            # STUB: For now, create a new track for every detection
            # Replace this with actual tracker association logic
            track = self._create_or_update_track(det, frame_detections.timestamp)
            if track.state == TrackState.ACTIVE:
                active.append(track)
            elif track.state == TrackState.OCCLUDED:
                occluded.append(track)
            elif track.state == TrackState.LOST:
                lost.append(track)

        # Age unmatched tracks
        self._age_unmatched_tracks(frame_detections.timestamp)

        # Collect lost tracks
        for tid, track in self.tracks.items():
            if track.state == TrackState.LOST and track not in lost:
                lost.append(track)

        return active, occluded, lost

    # ----------------------------------------------------------------
    # APPROACH B: Custom tracker with boxmot (for full project)
    # ----------------------------------------------------------------

    def _update_custom(self, frame_detections: FrameDetections) -> tuple[list, list, list]:
        """
        TODO (Dhruvish — post-MVP): Full custom tracking pipeline.

        Use boxmot library:
          from boxmot import BoTSORT  (or DeepOCSORT, ByteTrack)
          tracker = BoTSORT(reid_weights='osnet_x1_0_msmt17.pt', ...)
          tracks = tracker.update(dets, frame)

        This gives you direct access to:
          - Association cost matrix (IoU + appearance)
          - Per-track state management
          - Kalman filter internals for motion prediction
        """
        raise NotImplementedError("Custom tracker not yet implemented")

    # ----------------------------------------------------------------
    # TRACK MANAGEMENT
    # ----------------------------------------------------------------

    def _create_or_update_track(self, det, timestamp: float) -> Track:
        """
        STUB — Replace with proper association logic.

        Real implementation should:
          1. Compute IoU between detection and all active track predictions
          2. Compute appearance similarity if embeddings available
          3. Use Hungarian algorithm to find optimal assignment
          4. Create new tracks for unmatched detections
          5. Mark unmatched tracks as occluded
        """
        # Simple nearest-neighbor matching (placeholder)
        best_match = None
        best_iou = 0.3  # minimum threshold

        for tid, track in self.tracks.items():
            if track.state in (TrackState.ACTIVE, TrackState.OCCLUDED):
                iou = self._compute_iou(det.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = tid

        if best_match is not None:
            # Update existing track
            track = self.tracks[best_match]
            old_center = track.center
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.state = TrackState.ACTIVE
            track.frames_since_seen = 0
            track.last_seen_timestamp = timestamp
            track.total_frames_tracked += 1

            # Update trajectory
            cx, cy = track.center
            track.trajectory_history.append((cx, cy, timestamp))
            if len(track.trajectory_history) > self.config.trail_length:
                track.trajectory_history.pop(0)

            # Update velocity
            if old_center:
                track.velocity = np.array([cx - old_center[0], cy - old_center[1]])
                track.instantaneous_speed = np.linalg.norm(track.velocity)

            # Update heatmap
            hx = int(cx / 1280 * self.config.heatmap_resolution[0])
            hy = int(cy / 720 * self.config.heatmap_resolution[1])
            hx = np.clip(hx, 0, self.config.heatmap_resolution[0] - 1)
            hy = np.clip(hy, 0, self.config.heatmap_resolution[1] - 1)
            self.heatmap[hy, hx] += 1

            return track
        else:
            # Create new track
            color = generate_unique_color(self.next_track_id)
            cx, cy = det.center
            track = Track(
                track_id=self.next_track_id,
                state=TrackState.ACTIVE,
                bbox=det.bbox,
                confidence=det.confidence,
                class_name=det.class_name,
                color=color,
                first_seen_timestamp=timestamp,
                last_seen_timestamp=timestamp,
                trajectory_history=[(cx, cy, timestamp)],
            )
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1
            self.total_entries += 1
            return track

    def _age_unmatched_tracks(self, timestamp: float):
        """Transition unmatched tracks through the state machine."""
        to_delete = []
        for tid, track in self.tracks.items():
            if track.last_seen_timestamp < timestamp - 0.01:  # not updated this frame
                track.frames_since_seen += 1

                if track.state == TrackState.ACTIVE:
                    if track.frames_since_seen > 3:
                        track.state = TrackState.OCCLUDED
                        # Predict position using velocity
                        cx, cy = track.center
                        vx, vy = track.velocity
                        pred_cx = cx + vx * track.frames_since_seen
                        pred_cy = cy + vy * track.frames_since_seen
                        w = track.bbox[2] - track.bbox[0]
                        h = track.bbox[3] - track.bbox[1]
                        track.predicted_position = np.array([pred_cx, pred_cy])
                        track.bbox = np.array([pred_cx - w/2, pred_cy - h/2,
                                               pred_cx + w/2, pred_cy + h/2])

                elif track.state == TrackState.OCCLUDED:
                    if track.frames_since_seen > self.config.max_occlusion_frames:
                        track.state = TrackState.LOST

                elif track.state == TrackState.LOST:
                    if track.frames_since_seen > self.config.max_lost_frames:
                        to_delete.append(tid)
                        self.total_exits += 1

        for tid in to_delete:
            self.deleted_tracks[tid] = self.tracks.pop(tid)
            self.deleted_tracks[tid].state = TrackState.DELETED

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

    # ----------------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------------

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
