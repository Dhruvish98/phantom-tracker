"""
Visualization Module — Owner: AGASTYA
========================================
Renders all visual overlays on the frame and the analytics dashboard.

Layers (all toggleable via keyboard):
  [t] Trajectory trails — fading motion history
  [g] Ghost outlines — predicted position of occluded objects
  [p] Predicted path — dotted line showing future trajectory
  [i] ID labels — bounding box + ID + class
  [f] FPS counter

MVP target: Bounding boxes + IDs + trajectory trails + FPS counter.

TODO (Agastya):
  [ ] Get basic bbox + ID overlay working on tracker output
  [ ] Add trajectory trails with configurable fade
  [ ] Add FPS counter overlay
  [ ] Ghost outline rendering (post-MVP)
  [ ] Predicted future path (post-MVP)
  [ ] Analytics dashboard side panel (post-MVP)
"""

import cv2
import numpy as np
from core.interfaces import (
    FrameState, Track, TrackState, AnalyticsSnapshot, PipelineConfig
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Visualizer:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def render(self, frame: np.ndarray, state: FrameState,
               analytics: AnalyticsSnapshot, fps: float) -> np.ndarray:
        """
        Main render call. Draws all overlays on a copy of the frame.
        Called every frame by main.py.
        """
        output = frame.copy()

        # Layer 1: Trajectory trails (behind boxes)
        if self.config.show_trails:
            output = self._draw_trails(output, state.active_tracks + state.occluded_tracks)

        # Layer 2: Ghost outlines for occluded objects
        if self.config.show_ghost_outlines:
            output = self._draw_ghost_outlines(output, state.occluded_tracks)

        # Layer 3: Predicted future path
        if self.config.show_predicted_path:
            output = self._draw_predicted_paths(output, state.active_tracks)

        # Layer 4: Bounding boxes + IDs (on top)
        if self.config.show_ids:
            output = self._draw_boxes_and_ids(output, state.active_tracks)

        # Layer 5: Re-ID event notification
        if state.reid_results and state.reid_results.matches:
            output = self._draw_reid_notification(output, state.reid_results)

        # Layer 6: FPS and status bar
        if self.config.show_fps:
            output = self._draw_status_bar(output, state, fps)

        return output

    # ----------------------------------------------------------------
    # BOUNDING BOXES & IDS
    # ----------------------------------------------------------------

    def _draw_boxes_and_ids(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """
        Draw bounding boxes with unique colors and ID labels.

        AGASTYA: This is your first implementation target.
        Each track has a persistent .color tuple (R,G,B).
        Convert to BGR for OpenCV: color[::-1]
        """
        for track in tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            color_bgr = track.color[::-1]  # RGB → BGR for OpenCV

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

            # ID label with background
            label = f"#{track.track_id} {track.class_name}"
            if track.confidence > 0:
                label += f" {track.confidence:.0%}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_bgr, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    # ----------------------------------------------------------------
    # TRAJECTORY TRAILS
    # ----------------------------------------------------------------

    def _draw_trails(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """
        Draw fading trajectory trails for each track.

        AGASTYA: Each track.trajectory_history is a list of (cx, cy, timestamp).
        Draw lines between consecutive points, with increasing opacity
        from old (faint) to new (solid).
        """
        for track in tracks:
            pts = track.trajectory_history
            if len(pts) < 2:
                continue

            color_bgr = track.color[::-1]
            n = len(pts)

            for i in range(1, n):
                # Fade: older points are more transparent
                alpha = i / n  # 0.0 (oldest) to 1.0 (newest)
                thickness = max(1, int(alpha * 3))

                pt1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))

                # Simple approach: vary color intensity
                faded_color = tuple(int(c * alpha) for c in color_bgr)
                cv2.line(frame, pt1, pt2, faded_color, thickness)

        return frame

    # ----------------------------------------------------------------
    # GHOST OUTLINES (Post-MVP)
    # ----------------------------------------------------------------

    def _draw_ghost_outlines(self, frame: np.ndarray, occluded_tracks: list[Track]) -> np.ndarray:
        """
        Draw semi-transparent outlines at predicted position of occluded objects.

        TODO (Agastya — post-MVP):
          1. Get predicted bbox from track (Dhruvish provides this)
          2. Draw a dashed rectangle at that position
          3. Use cv2.addWeighted for transparency
          4. Add a "?" or ghost icon to indicate uncertainty
        """
        for track in occluded_tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            color_bgr = track.color[::-1]

            # Dashed rectangle (simple approach: draw dots along the edges)
            self._draw_dashed_rect(frame, (x1, y1), (x2, y2), color_bgr,
                                   thickness=2, dash_length=10)

            # Ghost label
            label = f"#{track.track_id} [occluded]"
            cv2.putText(frame, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)

        return frame

    def _draw_dashed_rect(self, frame, pt1, pt2, color, thickness=2, dash_length=10):
        """Draw a dashed rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        # Draw dashed lines for each edge
        for edge in [((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
                     ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))]:
            self._draw_dashed_line(frame, edge[0], edge[1], color, thickness, dash_length)

    def _draw_dashed_line(self, frame, pt1, pt2, color, thickness, dash_length):
        """Draw a dashed line between two points."""
        dist = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
        if dist == 0:
            return
        n_dashes = int(dist / dash_length)
        for i in range(0, n_dashes, 2):
            start = (int(pt1[0] + (pt2[0]-pt1[0]) * i/n_dashes),
                     int(pt1[1] + (pt2[1]-pt1[1]) * i/n_dashes))
            end = (int(pt1[0] + (pt2[0]-pt1[0]) * min(i+1, n_dashes)/n_dashes),
                   int(pt1[1] + (pt2[1]-pt1[1]) * min(i+1, n_dashes)/n_dashes))
            cv2.line(frame, start, end, color, thickness)

    # ----------------------------------------------------------------
    # PREDICTED PATHS (Post-MVP)
    # ----------------------------------------------------------------

    def _draw_predicted_paths(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """
        Draw dotted lines showing predicted future trajectory.

        TODO (Agastya — post-MVP):
          Each track.predicted_trajectory is a list of (cx, cy) future points.
          Draw dotted line from current position through predicted points.
        """
        for track in tracks:
            if not track.predicted_trajectory:
                continue

            color_bgr = track.color[::-1]
            pts = [track.center] + track.predicted_trajectory

            for i in range(1, len(pts)):
                pt1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                # Dotted line using dashed drawing
                self._draw_dashed_line(frame, pt1, pt2, color_bgr, 1, 6)

        return frame

    # ----------------------------------------------------------------
    # RE-ID NOTIFICATION
    # ----------------------------------------------------------------

    def _draw_reid_notification(self, frame: np.ndarray, reid_results) -> np.ndarray:
        """
        Show a notification when Re-ID matches someone.
        e.g., "Person #3 returned — confidence: 94.2%"
        """
        y_offset = 60
        for match in reid_results.matches:
            if match.is_confident:
                text = f"Re-ID: Track #{match.matched_track_id} returned ({match.similarity_score:.1%})"
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 30
        return frame

    # ----------------------------------------------------------------
    # STATUS BAR
    # ----------------------------------------------------------------

    def _draw_status_bar(self, frame: np.ndarray, state: FrameState, fps: float) -> np.ndarray:
        """Top-left status bar with FPS, track count, etc."""
        h, w = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        info = (f"FPS: {fps:.0f} | "
                f"Active: {len(state.active_tracks)} | "
                f"Occluded: {len(state.occluded_tracks)} | "
                f"Lost: {len(state.lost_tracks)}")

        cv2.putText(frame, info, (8, 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return frame
