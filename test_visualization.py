"""
Test script for Phantom Tracker Visualization Module
Owner: Agastya

This script tests all visualization features with synthetic data.
Run: python test_visualization.py
"""

import cv2
import numpy as np
import time
from core.interfaces import (
    FrameState, Track, TrackState, Detection, FrameDetections,
    AnalyticsSnapshot, PipelineConfig
)
from visualization.visualizer import Visualizer
from utils.fps_counter import FPSCounter
from utils.colors import generate_unique_color


def create_synthetic_frame(width=1280, height=720):
    """Create a synthetic frame with gradient background."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient background
    for y in range(height):
        intensity = int(30 + (y / height) * 50)
        frame[y, :] = [intensity, intensity, intensity]
    
    # Add grid lines
    for x in range(0, width, 100):
        cv2.line(frame, (x, 0), (x, height), (60, 60, 60), 1)
    for y in range(0, height, 100):
        cv2.line(frame, (0, y), (width, y), (60, 60, 60), 1)
    
    return frame


def create_synthetic_tracks(frame_id, timestamp):
    """Create synthetic tracks with various states for testing."""
    tracks = []
    
    # Active track 1: Moving right
    track1 = Track(
        track_id=1,
        state=TrackState.ACTIVE,
        bbox=np.array([100 + frame_id * 3, 200, 180 + frame_id * 3, 320]),
        confidence=0.95,
        class_name="person",
        color=generate_unique_color(1),
        velocity=np.array([3.0, 0.5]),
        instantaneous_speed=3.04,
        trajectory_history=[(100 + i * 3, 260, timestamp - (60-i)*0.033) 
                           for i in range(max(0, frame_id-60), frame_id)],
        frames_since_seen=0,
        total_frames_tracked=frame_id,
        first_seen_timestamp=timestamp - frame_id * 0.033,
        last_seen_timestamp=timestamp
    )
    
    # Generate predicted trajectory
    cx, cy = track1.center
    vx, vy = track1.velocity
    track1.predicted_trajectory = [(cx + vx * t, cy + vy * t) for t in range(5, 35, 5)]
    
    tracks.append(track1)
    
    # Active track 2: Moving diagonally
    track2 = Track(
        track_id=2,
        state=TrackState.ACTIVE,
        bbox=np.array([400 + frame_id * 2, 100 + frame_id * 1.5, 
                      480 + frame_id * 2, 220 + frame_id * 1.5]),
        confidence=0.88,
        class_name="person",
        color=generate_unique_color(2),
        velocity=np.array([2.0, 1.5]),
        instantaneous_speed=2.5,
        trajectory_history=[(400 + i * 2, 160 + i * 1.5, timestamp - (40-i)*0.033) 
                           for i in range(max(0, frame_id-40), frame_id)],
        frames_since_seen=0,
        total_frames_tracked=frame_id,
        first_seen_timestamp=timestamp - frame_id * 0.033,
        last_seen_timestamp=timestamp
    )
    track2.predicted_trajectory = [(track2.center[0] + 2*t, track2.center[1] + 1.5*t) 
                                   for t in range(5, 35, 5)]
    tracks.append(track2)
    
    # Occluded track 3
    if frame_id > 30:
        track3 = Track(
            track_id=3,
            state=TrackState.OCCLUDED,
            bbox=np.array([700, 400, 780, 520]),
            confidence=0.75,
            class_name="person",
            color=generate_unique_color(3),
            velocity=np.array([1.0, -0.5]),
            instantaneous_speed=1.12,
            trajectory_history=[(700 - i, 460 + i*0.5, timestamp - i*0.033) 
                               for i in range(20)],
            frames_since_seen=min(frame_id - 30, 15),
            total_frames_tracked=30,
            first_seen_timestamp=timestamp - 30 * 0.033,
            last_seen_timestamp=timestamp - (frame_id - 30) * 0.033
        )
        tracks.append(track3)
    
    # Fast moving track 4
    if frame_id > 20:
        track4 = Track(
            track_id=4,
            state=TrackState.ACTIVE,
            bbox=np.array([200 + frame_id * 5, 500, 260 + frame_id * 5, 600]),
            confidence=0.92,
            class_name="person",
            color=generate_unique_color(4),
            velocity=np.array([5.0, 0.2]),
            instantaneous_speed=5.0,
            trajectory_history=[(200 + i * 5, 550, timestamp - (20-i)*0.033) 
                               for i in range(max(0, frame_id-20), min(frame_id, 20))],
            frames_since_seen=0,
            total_frames_tracked=min(frame_id - 20, 20),
            first_seen_timestamp=timestamp - min(frame_id - 20, 20) * 0.033,
            last_seen_timestamp=timestamp
        )
        track4.predicted_trajectory = [(track4.center[0] + 5*t, track4.center[1] + 0.2*t) 
                                       for t in range(5, 35, 5)]
        tracks.append(track4)
    
    return tracks


def create_synthetic_analytics(frame_id, timestamp, tracks):
    """Create synthetic analytics data."""
    # Create heatmap
    heatmap = np.zeros((48, 64), dtype=np.float32)
    
    # Add heat for each track
    for track in tracks:
        if track.state == TrackState.ACTIVE:
            cx, cy = track.center
            hx = int(cx / 1280 * 64)
            hy = int(cy / 720 * 48)
            hx = np.clip(hx, 0, 63)
            hy = np.clip(hy, 0, 47)
            
            # Add gaussian blob
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = hx + dx, hy + dy
                    if 0 <= nx < 64 and 0 <= ny < 48:
                        dist = np.sqrt(dx**2 + dy**2)
                        heatmap[ny, nx] += np.exp(-dist / 2) * 10
    
    # Track speeds
    speeds = {t.track_id: t.instantaneous_speed for t in tracks 
              if t.state == TrackState.ACTIVE}
    
    # Dwell times
    dwell_times = {t.track_id: timestamp - t.first_seen_timestamp 
                   for t in tracks}
    
    return AnalyticsSnapshot(
        frame_id=frame_id,
        timestamp=timestamp,
        track_speeds=speeds,
        track_dwell_times=dwell_times,
        heatmap_accumulator=heatmap,
        total_entries=len(tracks) + 2,
        total_exits=1,
        current_object_count=len([t for t in tracks if t.state == TrackState.ACTIVE]),
        reid_events=["Track #3 re-identified (87.3%)", "Track #5 re-identified (92.1%)"]
    )


def main():
    """Run visualization test."""
    print("=" * 60)
    print("Phantom Tracker - Visualization Module Test")
    print("Owner: Agastya")
    print("=" * 60)
    print("\nKeyboard Controls:")
    print("  [Q] Quit")
    print("  [T] Toggle trajectory trails")
    print("  [G] Toggle ghost outlines")
    print("  [P] Toggle predicted paths")
    print("  [I] Toggle ID labels")
    print("  [F] Toggle FPS counter")
    print("  [H] Toggle heatmap overlay")
    print("  [D] Toggle analytics dashboard")
    print("\nStarting test...\n")
    
    # Initialize
    config = PipelineConfig()
    config.show_trails = True
    config.show_ghost_outlines = True
    config.show_predicted_path = True
    config.show_ids = True
    config.show_fps = True
    
    visualizer = Visualizer(config)
    visualizer.show_heatmap = False
    visualizer.show_dashboard = True
    
    fps_counter = FPSCounter()
    
    frame_id = 0
    start_time = time.time()
    
    try:
        while True:
            frame_id += 1
            timestamp = time.time()
            
            # Create synthetic data
            frame = create_synthetic_frame()
            tracks = create_synthetic_tracks(frame_id, timestamp)
            
            # Separate tracks by state
            active_tracks = [t for t in tracks if t.state == TrackState.ACTIVE]
            occluded_tracks = [t for t in tracks if t.state == TrackState.OCCLUDED]
            lost_tracks = [t for t in tracks if t.state == TrackState.LOST]
            
            # Create frame state
            state = FrameState(
                frame_id=frame_id,
                timestamp=timestamp,
                raw_frame=frame,
                active_tracks=active_tracks,
                occluded_tracks=occluded_tracks,
                lost_tracks=lost_tracks
            )
            
            # Create analytics
            analytics = create_synthetic_analytics(frame_id, timestamp, tracks)
            
            # Render
            output = visualizer.render(frame, state, analytics, fps_counter.get_fps())
            
            # Display
            cv2.imshow("Phantom Tracker - Visualization Test", output)
            
            # Handle keyboard
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                config.show_trails = not config.show_trails
                print(f"Trails: {config.show_trails}")
            elif key == ord('g'):
                config.show_ghost_outlines = not config.show_ghost_outlines
                print(f"Ghost outlines: {config.show_ghost_outlines}")
            elif key == ord('p'):
                config.show_predicted_path = not config.show_predicted_path
                print(f"Predicted paths: {config.show_predicted_path}")
            elif key == ord('i'):
                config.show_ids = not config.show_ids
                print(f"ID labels: {config.show_ids}")
            elif key == ord('f'):
                config.show_fps = not config.show_fps
                print(f"FPS counter: {config.show_fps}")
            elif key == ord('h'):
                visualizer.show_heatmap = not visualizer.show_heatmap
                print(f"Heatmap: {visualizer.show_heatmap}")
            elif key == ord('d'):
                visualizer.show_dashboard = not visualizer.show_dashboard
                print(f"Dashboard: {visualizer.show_dashboard}")
            
            fps_counter.tick()
            
            # Reset after 300 frames for continuous demo
            if frame_id >= 300:
                frame_id = 0
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"\nTest completed!")
        print(f"Average FPS: {fps_counter.get_avg_fps():.1f}")
        print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
