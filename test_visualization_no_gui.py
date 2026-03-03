"""
Test script for Phantom Tracker Visualization Module (No GUI)
Owner: Agastya

This script tests all visualization features without requiring GUI support.
Run: python test_visualization_no_gui.py
"""

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
    
    # Occluded track 2
    if frame_id > 30:
        track2 = Track(
            track_id=2,
            state=TrackState.OCCLUDED,
            bbox=np.array([700, 400, 780, 520]),
            confidence=0.75,
            class_name="person",
            color=generate_unique_color(2),
            velocity=np.array([1.0, -0.5]),
            instantaneous_speed=1.12,
            trajectory_history=[(700 - i, 460 + i*0.5, timestamp - i*0.033) 
                               for i in range(20)],
            frames_since_seen=min(frame_id - 30, 15),
            total_frames_tracked=30,
            first_seen_timestamp=timestamp - 30 * 0.033,
            last_seen_timestamp=timestamp - (frame_id - 30) * 0.033
        )
        tracks.append(track2)
    
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
    """Run visualization test without GUI."""
    print("=" * 60)
    print("Phantom Tracker - Visualization Module Test (No GUI)")
    print("Owner: Agastya")
    print("=" * 60)
    print("\nTesting all visualization features...")
    
    # Initialize
    config = PipelineConfig()
    config.show_trails = True
    config.show_ghost_outlines = True
    config.show_predicted_path = True
    config.show_ids = True
    config.show_fps = True
    
    visualizer = Visualizer(config)
    visualizer.show_heatmap = True
    visualizer.show_dashboard = True
    
    fps_counter = FPSCounter()
    
    # Test scenarios
    test_cases = [
        ("Basic rendering", 10),
        ("With occluded tracks", 50),
        ("All features enabled", 100),
    ]
    
    print("\nRunning test cases:")
    
    for test_name, frame_count in test_cases:
        print(f"\n  Testing: {test_name}")
        start_time = time.time()
        
        for frame_id in range(1, frame_count + 1):
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
            try:
                output = visualizer.render(frame, state, analytics, fps_counter.get_fps())
                
                # Verify output
                assert output is not None, "Output frame is None"
                assert output.shape == frame.shape, "Output shape mismatch"
                assert output.dtype == frame.dtype, "Output dtype mismatch"
                
            except Exception as e:
                print(f"    ❌ Error at frame {frame_id}: {e}")
                raise
            
            fps_counter.tick()
        
        elapsed = time.time() - start_time
        avg_fps = fps_counter.get_avg_fps()
        
        print(f"    ✅ Passed: {frame_count} frames in {elapsed:.2f}s ({avg_fps:.1f} FPS)")
    
    # Test individual features
    print("\n\nTesting individual features:")
    
    features = [
        ("Bounding boxes", {"show_ids": True}),
        ("Trajectory trails", {"show_trails": True}),
        ("Ghost outlines", {"show_ghost_outlines": True}),
        ("Predicted paths", {"show_predicted_path": True}),
        ("Heatmap overlay", {"show_heatmap": True}),
        ("Analytics dashboard", {"show_dashboard": True}),
    ]
    
    for feature_name, feature_config in features:
        # Reset config
        config = PipelineConfig()
        for key, value in feature_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(visualizer, key, value)
        
        # Test single frame
        frame = create_synthetic_frame()
        tracks = create_synthetic_tracks(50, time.time())
        
        state = FrameState(
            frame_id=50,
            timestamp=time.time(),
            raw_frame=frame,
            active_tracks=[t for t in tracks if t.state == TrackState.ACTIVE],
            occluded_tracks=[t for t in tracks if t.state == TrackState.OCCLUDED],
            lost_tracks=[]
        )
        
        analytics = create_synthetic_analytics(50, time.time(), tracks)
        
        try:
            output = visualizer.render(frame, state, analytics, 30.0)
            assert output is not None
            print(f"  ✅ {feature_name}")
        except Exception as e:
            print(f"  ❌ {feature_name}: {e}")
            raise
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print(f"\nVisualization module is working correctly!")
    print(f"Average FPS: {fps_counter.get_avg_fps():.1f}")
    print("\nTo test with GUI, install opencv-contrib-python:")
    print("  pip install opencv-contrib-python")
    print("\nThen run:")
    print("  python test_visualization.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
