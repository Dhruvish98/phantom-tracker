"""
Phantom Tracker - Visualization Showcase Demo
==============================================
Owner: Agastya

This demo showcases all visualization features with animated synthetic data.
Perfect for presentations and demonstrations.

Run: python demos/visualization_showcase.py
"""

import cv2
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import (
    FrameState, Track, TrackState, AnalyticsSnapshot, PipelineConfig, ReIDResult, ReIDMatch
)
from visualization.visualizer import Visualizer
from utils.fps_counter import FPSCounter
from utils.colors import generate_unique_color


class ShowcaseDemo:
    """Interactive visualization showcase."""
    
    def __init__(self):
        self.config = PipelineConfig()
        self.config.show_trails = True
        self.config.show_ghost_outlines = True
        self.config.show_predicted_path = True
        self.config.show_ids = True
        self.config.show_fps = True
        
        self.visualizer = Visualizer(self.config)
        self.visualizer.show_heatmap = True
        self.visualizer.show_dashboard = True
        
        self.fps_counter = FPSCounter()
        self.frame_id = 0
        self.start_time = time.time()
        
        # Demo scenarios
        self.scenarios = [
            "Basic Tracking",
            "Occlusion Handling",
            "Re-Identification",
            "Speed Visualization",
            "Heatmap Analysis"
        ]
        self.current_scenario = 0
        
    def create_background(self, width=1280, height=720):
        """Create professional background."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Gradient background
        for y in range(height):
            intensity = int(20 + (y / height) * 40)
            frame[y, :] = [intensity, intensity, intensity + 10]
        
        # Grid pattern
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (50, 50, 50), 1)
        for y in range(0, height, 80):
            cv2.line(frame, (0, y), (width, y), (50, 50, 50), 1)
        
        # Add title
        title = "PHANTOM TRACKER - VISUALIZATION SHOWCASE"
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(title, font, 1.2, 3)
        
        # Title background
        overlay = frame.copy()
        cv2.rectangle(overlay, (width//2 - tw//2 - 20, 20), 
                     (width//2 + tw//2 + 20, 70), (0, 100, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, title, (width//2 - tw//2, 55),
                   font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        
        return frame
    
    def create_scenario_tracks(self, scenario_id, frame_id, timestamp):
        """Create tracks based on scenario."""
        tracks = []
        
        if scenario_id == 0:  # Basic Tracking
            # Simple left-to-right movement
            for i in range(3):
                track = Track(
                    track_id=i+1,
                    state=TrackState.ACTIVE,
                    bbox=np.array([100 + frame_id*2 + i*200, 200 + i*100, 
                                  180 + frame_id*2 + i*200, 320 + i*100]),
                    confidence=0.90 + i*0.03,
                    class_name="person",
                    color=generate_unique_color(i+1),
                    velocity=np.array([2.0, 0.0]),
                    instantaneous_speed=2.0,
                    trajectory_history=[(100 + j*2 + i*200, 260 + i*100, 
                                       timestamp - (frame_id-j)*0.033) 
                                      for j in range(max(0, frame_id-50), frame_id)],
                    frames_since_seen=0,
                    total_frames_tracked=frame_id,
                    first_seen_timestamp=timestamp - frame_id * 0.033,
                    last_seen_timestamp=timestamp
                )
                track.predicted_trajectory = [(track.center[0] + 2*t, track.center[1]) 
                                             for t in range(5, 30, 5)]
                tracks.append(track)
        
        elif scenario_id == 1:  # Occlusion Handling
            # Active track
            track1 = Track(
                track_id=1,
                state=TrackState.ACTIVE,
                bbox=np.array([200 + frame_id*3, 300, 280 + frame_id*3, 420]),
                confidence=0.95,
                class_name="person",
                color=generate_unique_color(1),
                velocity=np.array([3.0, 0.0]),
                instantaneous_speed=3.0,
                trajectory_history=[(200 + i*3, 360, timestamp - (frame_id-i)*0.033) 
                                   for i in range(max(0, frame_id-60), frame_id)],
                frames_since_seen=0,
                total_frames_tracked=frame_id,
                first_seen_timestamp=timestamp - frame_id * 0.033,
                last_seen_timestamp=timestamp
            )
            tracks.append(track1)
            
            # Occluded track (appears after frame 30)
            if frame_id > 30:
                track2 = Track(
                    track_id=2,
                    state=TrackState.OCCLUDED,
                    bbox=np.array([600, 250, 680, 370]),
                    confidence=0.85,
                    class_name="person",
                    color=generate_unique_color(2),
                    velocity=np.array([1.5, 0.5]),
                    instantaneous_speed=1.58,
                    trajectory_history=[(600 - i*1.5, 310 - i*0.5, 
                                       timestamp - i*0.033) for i in range(20)],
                    frames_since_seen=frame_id - 30,
                    total_frames_tracked=30,
                    first_seen_timestamp=timestamp - 30 * 0.033,
                    last_seen_timestamp=timestamp - (frame_id - 30) * 0.033
                )
                tracks.append(track2)
        
        elif scenario_id == 2:  # Re-Identification
            # Multiple tracks with re-ID events
            for i in range(4):
                state = TrackState.ACTIVE if i < 3 else TrackState.OCCLUDED
                track = Track(
                    track_id=i+1,
                    state=state,
                    bbox=np.array([150 + i*250, 200 + (i%2)*200, 
                                  230 + i*250, 320 + (i%2)*200]),
                    confidence=0.88 + i*0.02,
                    class_name="person",
                    color=generate_unique_color(i+1),
                    velocity=np.array([1.5, 0.5]),
                    instantaneous_speed=1.58,
                    trajectory_history=[(150 + i*250 - j, 260 + (i%2)*200, 
                                       timestamp - j*0.033) for j in range(30)],
                    frames_since_seen=0 if state == TrackState.ACTIVE else 10,
                    total_frames_tracked=frame_id,
                    first_seen_timestamp=timestamp - frame_id * 0.033,
                    last_seen_timestamp=timestamp
                )
                tracks.append(track)
        
        elif scenario_id == 3:  # Speed Visualization
            # Tracks with different speeds
            speeds = [2.0, 7.0, 18.0]  # Slow, medium, fast
            for i, speed in enumerate(speeds):
                track = Track(
                    track_id=i+1,
                    state=TrackState.ACTIVE,
                    bbox=np.array([100 + frame_id*speed, 150 + i*200, 
                                  180 + frame_id*speed, 270 + i*200]),
                    confidence=0.92,
                    class_name="person",
                    color=generate_unique_color(i+1),
                    velocity=np.array([speed, 0.5]),
                    instantaneous_speed=speed,
                    trajectory_history=[(100 + j*speed, 210 + i*200, 
                                       timestamp - (frame_id-j)*0.033) 
                                      for j in range(max(0, frame_id-40), frame_id)],
                    frames_since_seen=0,
                    total_frames_tracked=frame_id,
                    first_seen_timestamp=timestamp - frame_id * 0.033,
                    last_seen_timestamp=timestamp
                )
                track.predicted_trajectory = [(track.center[0] + speed*t, 
                                              track.center[1] + 0.5*t) 
                                             for t in range(5, 25, 5)]
                tracks.append(track)
        
        elif scenario_id == 4:  # Heatmap Analysis
            # Circular motion for heatmap
            angle = frame_id * 0.05
            radius = 200
            for i in range(3):
                offset_angle = angle + i * (2 * np.pi / 3)
                cx = 640 + radius * np.cos(offset_angle)
                cy = 360 + radius * np.sin(offset_angle)
                
                track = Track(
                    track_id=i+1,
                    state=TrackState.ACTIVE,
                    bbox=np.array([cx-40, cy-60, cx+40, cy+60]),
                    confidence=0.90,
                    class_name="person",
                    color=generate_unique_color(i+1),
                    velocity=np.array([3*np.cos(offset_angle), 3*np.sin(offset_angle)]),
                    instantaneous_speed=3.0,
                    trajectory_history=[(640 + radius*np.cos(angle - j*0.05 + i*2*np.pi/3),
                                       360 + radius*np.sin(angle - j*0.05 + i*2*np.pi/3),
                                       timestamp - j*0.033) for j in range(60)],
                    frames_since_seen=0,
                    total_frames_tracked=frame_id,
                    first_seen_timestamp=timestamp - frame_id * 0.033,
                    last_seen_timestamp=timestamp
                )
                tracks.append(track)
        
        return tracks
    
    def create_analytics(self, frame_id, timestamp, tracks):
        """Create analytics data."""
        # Heatmap
        heatmap = np.zeros((48, 64), dtype=np.float32)
        for track in tracks:
            if track.state == TrackState.ACTIVE:
                for cx, cy, _ in track.trajectory_history:
                    hx = int(cx / 1280 * 64)
                    hy = int(cy / 720 * 48)
                    if 0 <= hx < 64 and 0 <= hy < 48:
                        heatmap[hy, hx] += 1
        
        return AnalyticsSnapshot(
            frame_id=frame_id,
            timestamp=timestamp,
            track_speeds={t.track_id: t.instantaneous_speed for t in tracks 
                         if t.state == TrackState.ACTIVE},
            track_dwell_times={t.track_id: timestamp - t.first_seen_timestamp 
                              for t in tracks},
            heatmap_accumulator=heatmap,
            total_entries=len(tracks) + 3,
            total_exits=2,
            current_object_count=len([t for t in tracks if t.state == TrackState.ACTIVE]),
            reid_events=["Track #2 re-identified (91.5%)", "Track #4 re-identified (88.2%)"]
        )
    
    def run(self):
        """Run the showcase demo."""
        print("=" * 70)
        print("PHANTOM TRACKER - VISUALIZATION SHOWCASE")
        print("=" * 70)
        print("\nScenarios:")
        for i, scenario in enumerate(self.scenarios):
            print(f"  {i+1}. {scenario}")
        print("\nControls:")
        print("  [SPACE] Next scenario")
        print("  [T/G/P/I/F/H/D] Toggle features")
        print("  [Q] Quit")
        print("\nStarting demo...\n")
        
        try:
            while True:
                self.frame_id += 1
                timestamp = time.time()
                
                # Create frame
                frame = self.create_background()
                
                # Add scenario label
                scenario_text = f"Scenario {self.current_scenario + 1}: {self.scenarios[self.current_scenario]}"
                cv2.putText(frame, scenario_text, (20, 100),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 200, 255), 2, cv2.LINE_AA)
                
                # Create tracks
                tracks = self.create_scenario_tracks(self.current_scenario, 
                                                    self.frame_id, timestamp)
                
                # Create state
                state = FrameState(
                    frame_id=self.frame_id,
                    timestamp=timestamp,
                    raw_frame=frame,
                    active_tracks=[t for t in tracks if t.state == TrackState.ACTIVE],
                    occluded_tracks=[t for t in tracks if t.state == TrackState.OCCLUDED],
                    lost_tracks=[t for t in tracks if t.state == TrackState.LOST]
                )
                
                # Add Re-ID results for scenario 2
                if self.current_scenario == 2 and self.frame_id % 90 == 0:
                    state.reid_results = ReIDResult(
                        frame_id=self.frame_id,
                        matches=[ReIDMatch(0, 2, 0.915, True)],
                        unmatched_detection_indices=[],
                        inference_time_ms=12.5
                    )
                
                # Create analytics
                analytics = self.create_analytics(self.frame_id, timestamp, tracks)
                
                # Render
                output = self.visualizer.render(frame, state, analytics, 
                                               self.fps_counter.get_fps())
                
                # Display
                cv2.imshow("Phantom Tracker - Visualization Showcase", output)
                
                # Handle keyboard
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.current_scenario = (self.current_scenario + 1) % len(self.scenarios)
                    self.frame_id = 0
                    print(f"Switched to: {self.scenarios[self.current_scenario]}")
                elif key == ord('t'):
                    self.config.show_trails = not self.config.show_trails
                elif key == ord('g'):
                    self.config.show_ghost_outlines = not self.config.show_ghost_outlines
                elif key == ord('p'):
                    self.config.show_predicted_path = not self.config.show_predicted_path
                elif key == ord('i'):
                    self.config.show_ids = not self.config.show_ids
                elif key == ord('f'):
                    self.config.show_fps = not self.config.show_fps
                elif key == ord('h'):
                    self.visualizer.show_heatmap = not self.visualizer.show_heatmap
                elif key == ord('d'):
                    self.visualizer.show_dashboard = not self.visualizer.show_dashboard
                
                self.fps_counter.tick()
                
                # Reset after 200 frames
                if self.frame_id >= 200:
                    self.frame_id = 0
                    
        except KeyboardInterrupt:
            print("\nDemo interrupted")
        finally:
            cv2.destroyAllWindows()
            print(f"\nDemo completed!")
            print(f"Average FPS: {self.fps_counter.get_avg_fps():.1f}")


if __name__ == "__main__":
    demo = ShowcaseDemo()
    demo.run()
