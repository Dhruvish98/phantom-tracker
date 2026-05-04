"""
Visualization Module — Owner: AGASTYA
========================================
Renders all visual overlays on the frame and the analytics dashboard.

Layers (all toggleable via keyboard):
  [t] Trajectory trails — fading motion history with smooth gradients
  [g] Ghost outlines — predicted position of occluded objects with transparency
  [p] Predicted path — dotted line showing future trajectory
  [i] ID labels — bounding box + ID + class with professional styling
  [f] FPS counter — performance metrics and system stats
  [h] Heatmap — density visualization overlay
  [d] Dashboard — analytics side panel with charts

Features:
  ✓ Professional bounding boxes with unique colors
  ✓ Smooth trajectory trails with alpha blending
  ✓ Ghost outlines for occluded objects
  ✓ Predicted future paths with uncertainty visualization
  ✓ Real-time analytics dashboard
  ✓ Heatmap overlay with color gradients
  ✓ Speed indicators and motion vectors
  ✓ Re-ID event notifications with animations
  ✓ Comprehensive status bar with metrics
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from core.interfaces import (
    FrameState, Track, TrackState, AnalyticsSnapshot, PipelineConfig
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Visualizer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.show_heatmap = False
        self.show_dashboard = False
        self.reid_notifications = []  # List of (message, timestamp, duration)
        self.notification_duration = 3.0  # seconds
        
        # Color maps for heatmap
        self.heatmap_colormap = cv2.COLORMAP_JET
        
        # Dashboard dimensions
        self.dashboard_width = 300
        self.dashboard_padding = 10
        
        logger.info("Visualizer initialized with advanced rendering features")

    def render(self, frame: np.ndarray, state: FrameState,
               analytics: AnalyticsSnapshot, fps: float) -> np.ndarray:
        """
        Main render call. Draws all overlays on a copy of the frame.
        Called every frame by main.py.
        
        Rendering order (back to front):
        1. Heatmap overlay (if enabled)
        2. Trajectory trails
        3. Ghost outlines for occluded objects
        4. Predicted future paths
        5. Bounding boxes + IDs
        6. Speed indicators
        7. Re-ID notifications
        8. Status bar
        9. Analytics dashboard (if enabled)
        """
        import time
        current_time = time.time()
        
        output = frame.copy()
        h, w = output.shape[:2]

        # Layer 0: Heatmap overlay (behind everything)
        if self.show_heatmap and analytics.heatmap_accumulator is not None:
            output = self._draw_heatmap_overlay(output, analytics.heatmap_accumulator)

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
            
        # Layer 5: Speed indicators
        output = self._draw_speed_indicators(output, state.active_tracks, analytics)

        # Layer 6: Re-ID event notification
        if state.reid_results and state.reid_results.matches:
            for match in state.reid_results.matches:
                if match.is_confident:
                    msg = f"Track #{match.matched_track_id} re-identified ({match.similarity_score:.1%})"
                    self.reid_notifications.append((msg, current_time, self.notification_duration))
        
        output = self._draw_reid_notifications(output, current_time)

        # Layer 7: FPS and status bar
        if self.config.show_fps:
            output = self._draw_status_bar(output, state, analytics, fps)
            
        # Layer 8: Analytics dashboard (side panel)
        if self.show_dashboard:
            output = self._draw_analytics_dashboard(output, state, analytics, fps)

        return output

    # ----------------------------------------------------------------
    # BOUNDING BOXES & IDS
    # ----------------------------------------------------------------

    def _draw_boxes_and_ids(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """
        Draw professional bounding boxes with unique colors and ID labels.
        
        Features:
        - Thick colored borders with subtle shadow effect
        - ID labels with semi-transparent background
        - Confidence scores
        - Corner markers for better visibility
        """
        for track in tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            color_bgr = track.color[::-1]  # RGB → BGR for OpenCV
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)

            # Draw shadow effect (offset by 2 pixels)
            shadow_color = (0, 0, 0)
            cv2.rectangle(frame, (x1+2, y1+2), (x2+2, y2+2), shadow_color, 2)
            
            # Main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)
            
            # Corner markers for better visibility
            corner_length = 15
            corner_thickness = 3
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color_bgr, corner_thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color_bgr, corner_thickness)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color_bgr, corner_thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color_bgr, corner_thickness)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color_bgr, corner_thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color_bgr, corner_thickness)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color_bgr, corner_thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color_bgr, corner_thickness)

            # ID label with background. In multi-camera mode (track.global_id set),
            # lead with the global identity so the same person reads the same on
            # both feeds; the per-camera track_id stays visible for debugging.
            if track.global_id is not None:
                label = f"G{track.global_id} (#{track.track_id}) {track.class_name}"
            else:
                label = f"ID:{track.track_id} {track.class_name}"
            if track.confidence > 0:
                label += f" {track.confidence:.0%}"

            # Calculate label size
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw semi-transparent background for label
            label_y = y1 - th - 10 if y1 - th - 10 > 0 else y2 + th + 10
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 4), 
                         color_bgr, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw text
            cv2.putText(frame, label, (x1 + 4, label_y),
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Draw center point
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 4, color_bgr, -1)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

        return frame

    # ----------------------------------------------------------------
    # TRAJECTORY TRAILS
    # ----------------------------------------------------------------

    def _draw_trails(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """
        Draw smooth fading trajectory trails with alpha blending.
        
        Features:
        - Smooth gradient from transparent (old) to opaque (new)
        - Variable thickness based on recency
        - Anti-aliased lines for professional appearance
        """
        # Create overlay for alpha blending
        overlay = frame.copy()
        
        for track in tracks:
            pts = track.trajectory_history
            if len(pts) < 2:
                continue

            color_bgr = track.color[::-1]
            n = len(pts)

            for i in range(1, n):
                # Calculate fade: older points are more transparent
                alpha = (i / n) ** 0.7  # Power curve for smoother fade
                thickness = max(1, int(alpha * 4))

                pt1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))

                # Interpolate color with alpha
                faded_color = tuple(int(c * alpha) for c in color_bgr)
                
                # Draw on overlay with anti-aliasing
                cv2.line(overlay, pt1, pt2, faded_color, thickness, cv2.LINE_AA)
                
                # Add glow effect for recent points
                if i > n * 0.8:  # Last 20% of trail
                    glow_color = tuple(min(255, int(c * 1.3)) for c in color_bgr)
                    cv2.line(overlay, pt1, pt2, glow_color, thickness + 2, cv2.LINE_AA)

        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        return frame

    # ----------------------------------------------------------------
    # GHOST OUTLINES
    # ----------------------------------------------------------------

    def _draw_ghost_outlines(self, frame: np.ndarray, occluded_tracks: list[Track]) -> np.ndarray:
        """
        Draw semi-transparent outlines at predicted position of occluded objects.
        
        Features:
        - Dashed rectangles with pulsing effect
        - Ghost icon indicator
        - Transparency based on occlusion duration
        - Uncertainty visualization
        """
        if not occluded_tracks:
            return frame
            
        import time
        current_time = time.time()
        
        overlay = frame.copy()
        
        # Default alpha value
        default_alpha = self.config.ghost_opacity
        
        for track in occluded_tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            color_bgr = track.color[::-1]
            
            # Calculate transparency based on how long occluded
            # More occluded = more transparent
            occlusion_factor = min(1.0, track.frames_since_seen / 30.0)
            track_alpha = self.config.ghost_opacity * (1.0 - occlusion_factor * 0.5)

            # Pulsing effect (subtle)
            pulse = 0.8 + 0.2 * np.sin(current_time * 3)
            track_alpha *= pulse

            # Draw filled rectangle with transparency
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            
            # Dashed border
            self._draw_dashed_rect(frame, (x1, y1), (x2, y2), color_bgr,
                                   thickness=3, dash_length=12)

            # Ghost icon (👻 approximation)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            ghost_size = 20
            # Draw simple ghost shape
            cv2.circle(frame, (cx, cy - 5), ghost_size, color_bgr, 2)
            cv2.ellipse(frame, (cx, cy + 10), (ghost_size, ghost_size//2), 
                       0, 0, 180, color_bgr, 2)
            
            # Eyes
            cv2.circle(frame, (cx - 7, cy - 8), 3, color_bgr, -1)
            cv2.circle(frame, (cx + 7, cy - 8), 3, color_bgr, -1)

            # Uncertainty indicator (expanding circle)
            uncertainty_radius = int(20 + track.frames_since_seen * 2)
            cv2.circle(frame, (cx, cy), uncertainty_radius, color_bgr, 1, cv2.LINE_AA)

            # Ghost label
            label = f"ID:{track.track_id} [OCCLUDED {track.frames_since_seen}f]"
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            
            # Semi-transparent background
            label_bg = overlay.copy()
            cv2.rectangle(label_bg, (x1, y1 - th - 10), (x1 + tw + 8, y1 - 2), 
                         color_bgr, -1)
            cv2.addWeighted(label_bg, 0.5, overlay, 0.5, 0, overlay)
            
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                       font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Apply transparency using default alpha
        cv2.addWeighted(overlay, default_alpha, frame, 1 - default_alpha, 0, frame)
        
        return frame

    def _draw_dashed_rect(self, frame, pt1, pt2, color, thickness=2, dash_length=10):
        """Draw a dashed rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        # Draw dashed lines for each edge
        for edge in [((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
                     ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))]:
            self._draw_dashed_line(frame, edge[0], edge[1], color, thickness, dash_length)

    # ----------------------------------------------------------------
    # ANALYTICS DASHBOARD
    # ----------------------------------------------------------------
    
    def _draw_analytics_dashboard(self, frame: np.ndarray, state: FrameState,
                                  analytics: AnalyticsSnapshot, fps: float) -> np.ndarray:
        """
        Draw comprehensive analytics dashboard as a side panel.
        
        Features:
        - Real-time statistics
        - Speed distribution chart
        - Dwell time chart
        - Track history
        - System metrics
        """
        h, w = frame.shape[:2]
        panel_w = self.dashboard_width
        pad = self.dashboard_padding
        
        # Create dashboard panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Divider line
        cv2.line(frame, (w - panel_w, 0), (w - panel_w, h), (100, 100, 100), 2)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        y_pos = 30
        
        # Title
        cv2.putText(frame, "ANALYTICS", (w - panel_w + pad, y_pos),
                   font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_pos += 40
        
        # Divider
        cv2.line(frame, (w - panel_w + pad, y_pos), (w - pad, y_pos), 
                (100, 100, 100), 1)
        y_pos += 20
        
        # Statistics section
        stats = [
            ("Frame", f"{state.frame_id}"),
            ("FPS", f"{fps:.1f}"),
            ("Active Tracks", f"{len(state.active_tracks)}"),
            ("Total Entries", f"{analytics.total_entries}"),
            ("Total Exits", f"{analytics.total_exits}"),
            ("Current Count", f"{analytics.current_object_count}"),
        ]
        
        for label, value in stats:
            cv2.putText(frame, label, (w - panel_w + pad, y_pos),
                       font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(frame, value, (w - pad - 60, y_pos),
                       font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 25
        
        y_pos += 10
        cv2.line(frame, (w - panel_w + pad, y_pos), (w - pad, y_pos), 
                (100, 100, 100), 1)
        y_pos += 25
        
        # Speed distribution
        cv2.putText(frame, "SPEED DISTRIBUTION", (w - panel_w + pad, y_pos),
                   font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y_pos += 25
        
        if analytics.track_speeds:
            speeds = list(analytics.track_speeds.values())
            max_speed = max(speeds) if speeds else 1.0
            
            # Draw mini bar chart
            bar_width = (panel_w - 2 * pad) // len(speeds) if speeds else 10
            bar_x = w - panel_w + pad
            
            for i, speed in enumerate(speeds[:10]):  # Show max 10 tracks
                bar_height = int((speed / max_speed) * 60) if max_speed > 0 else 0
                bar_color = (0, 255, 0) if speed < 5 else (0, 255, 255) if speed < 15 else (0, 0, 255)
                
                cv2.rectangle(frame, 
                            (bar_x + i * bar_width, y_pos + 60 - bar_height),
                            (bar_x + (i + 1) * bar_width - 2, y_pos + 60),
                            bar_color, -1)
            
            y_pos += 70
        else:
            cv2.putText(frame, "No active tracks", (w - panel_w + pad + 10, y_pos + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            y_pos += 50
        
        # Dwell time section
        cv2.line(frame, (w - panel_w + pad, y_pos), (w - pad, y_pos), 
                (100, 100, 100), 1)
        y_pos += 25
        
        cv2.putText(frame, "TOP DWELL TIMES", (w - panel_w + pad, y_pos),
                   font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y_pos += 25
        
        if analytics.track_dwell_times:
            # Sort by dwell time
            sorted_dwell = sorted(analytics.track_dwell_times.items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
            
            for track_id, dwell_time in sorted_dwell:
                dwell_text = f"ID {track_id}: {dwell_time:.1f}s"
                cv2.putText(frame, dwell_text, (w - panel_w + pad + 10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
                y_pos += 22
        else:
            cv2.putText(frame, "No data", (w - panel_w + pad + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            y_pos += 25
        
        # Re-ID events section
        if analytics.reid_events:
            y_pos += 10
            cv2.line(frame, (w - panel_w + pad, y_pos), (w - pad, y_pos), 
                    (100, 100, 100), 1)
            y_pos += 25
            
            cv2.putText(frame, "RECENT RE-ID", (w - panel_w + pad, y_pos),
                       font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            y_pos += 25
            
            for event in analytics.reid_events[-3:]:  # Show last 3
                cv2.putText(frame, event, (w - panel_w + pad + 10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                y_pos += 20
        
        return frame
    
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
    # PREDICTED PATHS
    # ----------------------------------------------------------------

    def _draw_predicted_paths(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """
        Draw dotted lines showing predicted future trajectory with uncertainty cone.
        
        Features:
        - Dotted prediction line
        - Uncertainty cone (widens with distance)
        - Endpoint marker
        - Fade effect for far predictions
        """
        for track in tracks:
            if not track.predicted_trajectory and len(track.velocity) > 0:
                # Generate simple linear prediction if not provided
                cx, cy = track.center
                vx, vy = track.velocity
                
                # Predict next 30 frames (1 second at 30fps)
                track.predicted_trajectory = []
                for t in range(1, 31, 3):
                    pred_x = cx + vx * t
                    pred_y = cy + vy * t
                    track.predicted_trajectory.append((pred_x, pred_y))
            
            if not track.predicted_trajectory:
                continue

            color_bgr = track.color[::-1]
            pts = [track.center] + track.predicted_trajectory
            
            # Draw uncertainty cone
            overlay = frame.copy()
            for i in range(1, len(pts)):
                # Cone widens with distance
                uncertainty = i * 3
                pt = (int(pts[i][0]), int(pts[i][1]))
                alpha = max(0.1, 1.0 - i / len(pts))
                
                # Draw uncertainty circle
                cone_color = tuple(int(c * alpha) for c in color_bgr)
                cv2.circle(overlay, pt, uncertainty, cone_color, -1)
            
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            # Draw prediction line
            for i in range(1, len(pts)):
                pt1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                
                # Fade with distance
                alpha = max(0.3, 1.0 - i / len(pts))
                line_color = tuple(int(c * alpha) for c in color_bgr)
                
                # Dotted line
                self._draw_dashed_line(frame, pt1, pt2, line_color, 2, 8)
            
            # Draw endpoint marker
            if len(pts) > 1:
                end_pt = (int(pts[-1][0]), int(pts[-1][1]))
                cv2.circle(frame, end_pt, 6, color_bgr, 2)
                cv2.circle(frame, end_pt, 3, (255, 255, 255), -1)
                
                # Arrow pointing to endpoint
                if len(pts) > 2:
                    prev_pt = (int(pts[-2][0]), int(pts[-2][1]))
                    cv2.arrowedLine(frame, prev_pt, end_pt, color_bgr, 2, 
                                   tipLength=0.3)

        return frame

    # ----------------------------------------------------------------
    # SPEED INDICATORS
    # ----------------------------------------------------------------
    
    def _draw_speed_indicators(self, frame: np.ndarray, tracks: list[Track], 
                               analytics: AnalyticsSnapshot) -> np.ndarray:
        """
        Draw speed indicators and motion vectors for active tracks.
        
        Features:
        - Motion vector arrows
        - Speed value display
        - Color-coded speed (green=slow, yellow=medium, red=fast)
        """
        for track in tracks:
            if track.instantaneous_speed < 1.0:  # Skip stationary objects
                continue
                
            cx, cy = int(track.center[0]), int(track.center[1])
            vx, vy = track.velocity
            
            # Scale velocity for visualization
            scale = 3.0
            end_x = int(cx + vx * scale)
            end_y = int(cy + vy * scale)
            
            # Color based on speed (green -> yellow -> red)
            speed = track.instantaneous_speed
            if speed < 5:
                arrow_color = (0, 255, 0)  # Green
            elif speed < 15:
                arrow_color = (0, 255, 255)  # Yellow
            else:
                arrow_color = (0, 0, 255)  # Red
            
            # Draw motion vector
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), arrow_color, 
                           2, tipLength=0.3, line_type=cv2.LINE_AA)
            
            # Speed label
            speed_label = f"{speed:.1f} px/f"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(speed_label, font, 0.4, 1)
            
            # Position label near the arrow tip
            label_x = end_x + 5
            label_y = end_y
            
            # Background for readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (label_x - 2, label_y - th - 2), 
                         (label_x + tw + 2, label_y + 2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            cv2.putText(frame, speed_label, (label_x, label_y),
                       font, 0.4, arrow_color, 1, cv2.LINE_AA)
        
        return frame
    
    # ----------------------------------------------------------------
    # HEATMAP OVERLAY
    # ----------------------------------------------------------------
    
    def _draw_heatmap_overlay(self, frame: np.ndarray, 
                             heatmap: np.ndarray) -> np.ndarray:
        """
        Draw density heatmap overlay showing where objects spend most time.
        
        Features:
        - Color-mapped density visualization
        - Transparency blending
        - Smooth interpolation
        """
        h, w = frame.shape[:2]
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            return frame
        
        # Resize to frame dimensions
        heatmap_resized = cv2.resize(heatmap_norm, (w, h), 
                                     interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, self.heatmap_colormap)
        
        # Blend with frame
        alpha = 0.4
        cv2.addWeighted(heatmap_colored, alpha, frame, 1 - alpha, 0, frame)
        
        # Add legend
        legend_h = 20
        legend_w = 200
        legend_x = w - legend_w - 20
        legend_y = h - legend_h - 20
        
        # Create gradient legend
        gradient = np.linspace(0, 255, legend_w).astype(np.uint8)
        gradient = np.tile(gradient, (legend_h, 1))
        gradient_colored = cv2.applyColorMap(gradient, self.heatmap_colormap)
        
        # Draw legend
        frame[legend_y:legend_y+legend_h, legend_x:legend_x+legend_w] = gradient_colored
        cv2.rectangle(frame, (legend_x, legend_y), 
                     (legend_x+legend_w, legend_y+legend_h), (255, 255, 255), 1)
        
        # Legend labels
        cv2.putText(frame, "Low", (legend_x - 30, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "High", (legend_x + legend_w + 5, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

    # ----------------------------------------------------------------
    # RE-ID NOTIFICATIONS
    # ----------------------------------------------------------------

    def _draw_reid_notifications(self, frame: np.ndarray, 
                                 current_time: float) -> np.ndarray:
        """
        Show animated notifications when Re-ID matches occur.
        
        Features:
        - Slide-in animation
        - Auto-fade after duration
        - Multiple notifications stacked
        """
        # Remove expired notifications
        self.reid_notifications = [
            (msg, ts, dur) for msg, ts, dur in self.reid_notifications
            if current_time - ts < dur
        ]
        
        y_offset = 60
        for msg, timestamp, duration in self.reid_notifications:
            elapsed = current_time - timestamp
            
            # Slide-in animation (first 0.3 seconds)
            if elapsed < 0.3:
                slide_progress = elapsed / 0.3
                x_offset = int((1 - slide_progress) * 300)
            else:
                x_offset = 0
            
            # Fade-out animation (last 0.5 seconds)
            if elapsed > duration - 0.5:
                fade_progress = (duration - elapsed) / 0.5
                alpha = fade_progress
            else:
                alpha = 1.0
            
            # Draw notification box
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = cv2.getTextSize(msg, font, 0.6, 2)
            
            box_x = 10 + x_offset
            box_y = y_offset
            box_w = tw + 20
            box_h = th + 16
            
            # Background with transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h),
                         (0, 200, 0), -1)
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h),
                         (0, 255, 0), 2)
            cv2.addWeighted(overlay, alpha * 0.8, frame, 1 - alpha * 0.8, 0, frame)
            
            # Text
            cv2.putText(frame, msg, (box_x + 10, box_y + th + 8),
                       font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Icon (checkmark)
            check_x = box_x + box_w - 25
            check_y = box_y + box_h // 2
            cv2.line(frame, (check_x, check_y), (check_x + 5, check_y + 5),
                    (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (check_x + 5, check_y + 5), (check_x + 12, check_y - 8),
                    (255, 255, 255), 2, cv2.LINE_AA)
            
            y_offset += box_h + 10
        
        return frame

    # ----------------------------------------------------------------
    # STATUS BAR
    # ----------------------------------------------------------------

    def _draw_status_bar(self, frame: np.ndarray, state: FrameState, 
                        analytics: AnalyticsSnapshot, fps: float) -> np.ndarray:
        """
        Professional status bar with comprehensive metrics.
        
        Features:
        - FPS with color coding (green=good, yellow=ok, red=poor)
        - Track counts by state
        - Entry/exit statistics
        - Detection timing
        - Keyboard shortcuts reminder
        """
        h, w = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        bar_height = 80
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Divider line
        cv2.line(frame, (0, bar_height), (w, bar_height), (100, 100, 100), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        
        # Row 1: FPS and performance
        fps_color = (0, 255, 0) if fps > 25 else (0, 255, 255) if fps > 15 else (0, 0, 255)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 25), font, 0.7, fps_color, 2, cv2.LINE_AA)
        
        # Detection time
        if state.detections:
            det_time = f"Det: {state.detections.inference_time_ms:.1f}ms"
            cv2.putText(frame, det_time, (150, 25), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Row 2: Track statistics
        active_text = f"Active: {len(state.active_tracks)}"
        occluded_text = f"Occluded: {len(state.occluded_tracks)}"
        lost_text = f"Lost: {len(state.lost_tracks)}"
        
        cv2.putText(frame, active_text, (10, 50), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, occluded_text, (120, 50), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, lost_text, (260, 50), font, 0.5, (100, 100, 255), 1, cv2.LINE_AA)
        
        # Row 3: Entry/Exit statistics
        entry_text = f"Entries: {analytics.total_entries}"
        exit_text = f"Exits: {analytics.total_exits}"
        current_text = f"Current: {analytics.current_object_count}"
        
        cv2.putText(frame, entry_text, (10, 72), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, exit_text, (120, 72), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, current_text, (220, 72), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Right side: Keyboard shortcuts
        shortcuts = [
            ("[Q] Quit", (255, 100, 100)),
            ("[T] Trails", (100, 255, 100) if self.config.show_trails else (100, 100, 100)),
            ("[G] Ghost", (100, 255, 100) if self.config.show_ghost_outlines else (100, 100, 100)),
            ("[P] Predict", (100, 255, 100) if self.config.show_predicted_path else (100, 100, 100)),
            ("[I] IDs", (100, 255, 100) if self.config.show_ids else (100, 100, 100)),
            ("[H] Heatmap", (100, 255, 100) if self.show_heatmap else (100, 100, 100)),
            ("[D] Dashboard", (100, 255, 100) if self.show_dashboard else (100, 100, 100)),
        ]
        
        x_start = w - 600
        y_pos = 25
        for i, (text, color) in enumerate(shortcuts):
            x_pos = x_start + (i % 4) * 150
            if i == 4:
                y_pos = 50
            cv2.putText(frame, text, (x_pos, y_pos), font, 0.4, color, 1, cv2.LINE_AA)
        
        return frame
