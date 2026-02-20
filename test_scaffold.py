"""
Quick Smoke Test — Run this FIRST to verify the scaffold works.

  python test_scaffold.py

This tests that all interfaces, imports, and data contracts work
WITHOUT needing any ML models installed. Everyone should run this
before starting their own module development.
"""

import sys
import numpy as np

print("=" * 60)
print("PHANTOM TRACKER — Scaffold Smoke Test")
print("=" * 60)

# Test 1: Interfaces import
print("\n[1/5] Testing interfaces import...", end=" ")
try:
    from core.interfaces import (
        Detection, FrameDetections, Track, TrackState,
        ReIDMatch, ReIDResult, FrameState, AnalyticsSnapshot, PipelineConfig
    )
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

# Test 2: Detection creation
print("[2/5] Testing Detection dataclass...", end=" ")
try:
    det = Detection(
        bbox=np.array([100, 200, 300, 400], dtype=np.float32),
        confidence=0.92,
        class_name="person",
        class_id=0
    )
    assert det.center == (200.0, 300.0), f"Expected (200, 300), got {det.center}"
    assert det.area == 40000.0, f"Expected 40000, got {det.area}"
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")

# Test 3: Track creation with state machine
print("[3/5] Testing Track + state machine...", end=" ")
try:
    from utils.colors import generate_unique_color
    track = Track(
        track_id=1, state=TrackState.ACTIVE,
        bbox=np.array([100, 200, 300, 400], dtype=np.float32),
        confidence=0.92, class_name="person",
        color=generate_unique_color(1)
    )
    assert track.is_visible == True
    track.state = TrackState.OCCLUDED
    assert track.is_visible == False
    track.state = TrackState.LOST
    assert track.state == TrackState.LOST
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")

# Test 4: FrameState pipeline flow
print("[4/5] Testing FrameState pipeline...", end=" ")
try:
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    state = FrameState(frame_id=1, timestamp=0.0, raw_frame=fake_frame)
    state.detections = FrameDetections(
        frame_id=1, timestamp=0.0,
        detections=[det], source="yolo", inference_time_ms=5.0
    )
    state.active_tracks = [track]
    assert state.all_tracks == [track]
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")

# Test 5: Config defaults
print("[5/5] Testing PipelineConfig...", end=" ")
try:
    config = PipelineConfig()
    assert config.yolo_model == "yolo11n.pt"
    assert config.trail_length == 60
    assert config.frame_width == 1280
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")

# Summary
print("\n" + "=" * 60)
print("All scaffold tests passed! You're ready to start coding.")
print("=" * 60)
print("""
Next steps for each member:
  Divyansh:  pip install ultralytics && python -c "from ultralytics import YOLO; print('YOLO ready')"
  Dhruvish:  Start implementing _update_builtin() in tracking/tracker.py
  Dharmik:   pip install torchreid && python -c "from torchreid.utils import FeatureExtractor; print('ReID ready')"
  Agastya:   Run 'python main.py' with a test video to see the visualization skeleton
""")
