from __future__ import annotations
"""
test_reid.py — Standalone Re-ID Module Tests
=============================================
Tests Dharmik's Re-ID module independently from the full pipeline.
Run this BEFORE integrating with Dhruvish's tracker.

Tests:
  1. Model loading (torchreid + OSNet)
  2. Embedding extraction from a dummy crop
  3. Feature bank add/similarity
  4. Full match() pipeline with mock tracks and detections
  5. Re-ID scenario: person leaves → returns → gets matched

Usage:
    python test_reid.py
    python test_reid.py --verbose
"""

import sys
import time
import argparse
import numpy as np

# Add project root to path so imports resolve
sys.path.insert(0, '.')

PASSED = 0
FAILED = 0


def test(name: str, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  [PASS] {name}")
        PASSED += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        FAILED += 1


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_config(**overrides):
    from core.interfaces import PipelineConfig
    cfg = PipelineConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_track(track_id: int, bbox=None, state=None, n_gallery: int = 5):
    """Create a Track with a realistic appearance gallery."""
    from core.interfaces import Track, TrackState
    bbox  = bbox or np.array([100., 100., 200., 400.])
    state = state or TrackState.LOST
    track = Track(
        track_id=track_id,
        state=state,
        bbox=bbox,
        confidence=0.9,
        class_name="person",
        color=(255, 0, 0),
    )
    # Fill gallery with random-but-consistent embeddings
    rng = np.random.default_rng(seed=track_id)
    base_emb = rng.standard_normal(512).astype(np.float32)
    base_emb /= np.linalg.norm(base_emb)
    for i in range(n_gallery):
        # Slight noise per entry to simulate real gallery
        noise = rng.standard_normal(512).astype(np.float32) * 0.05
        emb = base_emb + noise
        emb /= np.linalg.norm(emb)
        track.appearance_gallery.append((emb, time.time() - (n_gallery - i)))
    track.last_embedding = base_emb
    return track, base_emb


def make_frame_detections(frame_id: int, n: int = 3, embeddings=None):
    """Create a FrameDetections with n detections."""
    from core.interfaces import Detection, FrameDetections
    dets = []
    for i in range(n):
        emb = embeddings[i] if (embeddings and i < len(embeddings)) else None
        dets.append(Detection(
            bbox=np.array([50. + i*100, 100., 150. + i*100, 400.]),
            confidence=0.85,
            class_name="person",
            class_id=0,
            embedding=emb,
        ))
    return FrameDetections(frame_id=frame_id, timestamp=time.time(), detections=dets)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_feature_bank_add_and_similarity():
    from reid.reidentifier import FeatureBank
    from core.interfaces import Track, TrackState

    bank  = FeatureBank(max_size=10, decay_rate=0.9)
    track = Track(track_id=1, state=TrackState.ACTIVE,
                  bbox=np.zeros(4), confidence=1.0,
                  class_name="person", color=(255,0,0))

    # Create a base embedding and a similar query
    rng       = np.random.default_rng(42)
    base_emb  = rng.standard_normal(512).astype(np.float32)
    base_emb /= np.linalg.norm(base_emb)

    # Add 5 entries to gallery (slight noise each time)
    for _ in range(5):
        noise = rng.standard_normal(512).astype(np.float32) * 0.02
        e = base_emb + noise
        e /= np.linalg.norm(e)
        bank.add(track, e)

    assert len(track.appearance_gallery) == 5

    # Similar query should score high
    query = base_emb + rng.standard_normal(512).astype(np.float32) * 0.05
    query /= np.linalg.norm(query)
    sim = bank.similarity(query, track)
    assert sim > 0.5, f"Expected sim > 0.5, got {sim:.3f}"

    # Similar query should score higher than a random dissimilar query
    random_query = rng.standard_normal(512).astype(np.float32)
    random_query /= np.linalg.norm(random_query)
    low_sim = bank.similarity(random_query, track)
    assert low_sim < sim, f"Similar query ({sim:.3f}) should outscore random ({low_sim:.3f})"


def test_feature_bank_max_size():
    from reid.reidentifier import FeatureBank
    from core.interfaces import Track, TrackState

    bank  = FeatureBank(max_size=5, decay_rate=0.9)
    track = Track(track_id=1, state=TrackState.ACTIVE,
                  bbox=np.zeros(4), confidence=1.0,
                  class_name="person", color=(0,0,0))

    rng = np.random.default_rng(0)
    for _ in range(10):
        e = rng.standard_normal(512).astype(np.float32)
        bank.add(track, e)

    assert len(track.appearance_gallery) == 5, \
        f"Gallery should be capped at 5, got {len(track.appearance_gallery)}"


def test_feature_bank_best_match_correct_track():
    """The correct track should be matched when query embedding is similar."""
    from reid.reidentifier import FeatureBank

    bank = FeatureBank(max_size=20, decay_rate=0.95)
    rng  = np.random.default_rng(99)

    # Create 3 lost tracks with distinct embeddings
    tracks = []
    base_embs = []
    for tid in range(1, 4):
        track, base_emb = make_track(tid, n_gallery=10)
        tracks.append(track)
        base_embs.append(base_emb)

    # Query that matches track 2 (index 1)
    target_emb = base_embs[1].copy()
    noise = rng.standard_normal(512).astype(np.float32) * 0.03
    query = target_emb + noise
    query /= np.linalg.norm(query)

    best_id, score = bank.best_match(query, tracks, exclude_ids=set(), threshold=0.3)
    assert best_id == 2, f"Expected track_id=2, got {best_id} (score={score:.3f})"
    assert score > 0.4,  f"Expected score > 0.4, got {score:.3f}"


def test_feature_bank_excludes_claimed_ids():
    from reid.reidentifier import FeatureBank

    bank   = FeatureBank()
    track, base_emb = make_track(1, n_gallery=5)

    # Query matches track 1, but track 1 is in exclude set
    best_id, _ = bank.best_match(
        base_emb, [track], exclude_ids={1}, threshold=0.0
    )
    assert best_id == -1, "Should return -1 when best track is excluded"


def test_reidentifier_init_no_model():
    """ReIdentifier should initialise gracefully even without torchreid."""
    cfg = make_config(reid_model="osnet_x1_0")
    from reid.reidentifier import ReIdentifier
    reid = ReIdentifier(cfg)
    # model may be None if torchreid not installed — that's fine
    assert reid.bank is not None


def test_match_no_model_returns_empty():
    """match() with no model should return all detections as unmatched."""
    cfg  = make_config()
    from reid.reidentifier import ReIdentifier
    reid = ReIdentifier(cfg)
    reid.model = None   # simulate missing model

    track, _ = make_track(1)
    fd       = make_frame_detections(frame_id=42, n=3)
    result   = reid.match(fd, [track], active_track_ids=set())

    assert result.frame_id == 42
    assert len(result.matches) == 0
    assert result.unmatched_detection_indices == [0, 1, 2]


def test_match_with_precomputed_embeddings():
    """
    match() should correctly match a detection to a lost track
    when the detection already carries an embedding (from detector).
    """
    cfg  = make_config(reid_confidence_threshold=0.4)
    from reid.reidentifier import ReIdentifier
    reid = ReIdentifier(cfg)
    reid.model = None   # no model — we'll use precomputed embeddings

    # Create a lost track with a known embedding gallery
    track, base_emb = make_track(track_id=7, n_gallery=8)

    # Build a detection whose embedding is very similar to track 7
    rng   = np.random.default_rng(555)
    noise = rng.standard_normal(512).astype(np.float32) * 0.03
    query_emb  = base_emb + noise
    query_emb /= np.linalg.norm(query_emb)

    fd = make_frame_detections(frame_id=100, n=1, embeddings=[query_emb])
    result = reid.match(fd, [track], active_track_ids=set())

    assert len(result.matches) == 1,                "Should have 1 match"
    assert result.matches[0].matched_track_id == 7, "Should match track 7"
    assert result.matches[0].similarity_score > 0.4
    assert result.matches[0].is_confident is True
    assert result.unmatched_detection_indices == []


def test_match_rejects_below_threshold():
    """A detection with low similarity should NOT be matched."""
    cfg  = make_config(reid_confidence_threshold=0.9)   # very high threshold
    from reid.reidentifier import ReIdentifier
    reid = ReIdentifier(cfg)
    reid.model = None

    track, base_emb = make_track(track_id=3, n_gallery=5)

    # Completely random query — low similarity
    rng  = np.random.default_rng(11)
    rand = rng.standard_normal(512).astype(np.float32)
    rand /= np.linalg.norm(rand)

    fd     = make_frame_detections(frame_id=1, n=1, embeddings=[rand])
    result = reid.match(fd, [track], active_track_ids=set())

    assert len(result.matches) == 0
    assert 0 in result.unmatched_detection_indices


def test_match_no_double_assignment():
    """
    Two detections should NOT both match the same lost track.
    The second one should be unmatched.
    """
    cfg  = make_config(reid_confidence_threshold=0.5)
    from reid.reidentifier import ReIdentifier
    reid = ReIdentifier(cfg)
    reid.model = None

    track, base_emb = make_track(track_id=5, n_gallery=10)

    rng = np.random.default_rng(77)
    emb1 = base_emb + rng.standard_normal(512).astype(np.float32) * 0.02
    emb1 /= np.linalg.norm(emb1)
    emb2 = base_emb + rng.standard_normal(512).astype(np.float32) * 0.02
    emb2 /= np.linalg.norm(emb2)

    fd     = make_frame_detections(frame_id=5, n=2, embeddings=[emb1, emb2])
    result = reid.match(fd, [track], active_track_ids=set())

    matched_ids = [m.matched_track_id for m in result.matches]
    assert matched_ids.count(5) <= 1, "Track 5 should only be matched once"


def test_reid_scenario_person_leaves_and_returns():
    """
    MIDTERM DEMO SCENARIO:
    Person is tracked (gallery built), leaves frame → LOST state,
    then returns → Re-ID should recover the correct track ID.
    """
    cfg   = make_config(reid_confidence_threshold=0.4)
    from reid.reidentifier import ReIdentifier, FeatureBank
    from core.interfaces import TrackState

    reid  = ReIdentifier(cfg)
    reid.model = None   # test logic without GPU

    rng = np.random.default_rng(2024)

    # --- Phase 1: Person tracked for 30 frames, gallery built ---
    person_emb = rng.standard_normal(512).astype(np.float32)
    person_emb /= np.linalg.norm(person_emb)

    track, _ = make_track(track_id=42, n_gallery=0)   # empty gallery
    for frame_i in range(30):
        noise = rng.standard_normal(512).astype(np.float32) * 0.04
        emb   = person_emb + noise
        emb  /= np.linalg.norm(emb)
        reid.bank.add(track, emb)   # simulate gallery building during tracking

    assert len(track.appearance_gallery) == 30

    # --- Phase 2: Person leaves frame (track → LOST) ---
    track.state = TrackState.LOST

    # --- Phase 3: Person returns 8 seconds later, new detection ---
    time.sleep(0.01)   # tiny sleep to differentiate timestamps
    return_noise = rng.standard_normal(512).astype(np.float32) * 0.06
    return_emb   = person_emb + return_noise
    return_emb  /= np.linalg.norm(return_emb)

    fd     = make_frame_detections(frame_id=200, n=1, embeddings=[return_emb])
    result = reid.match(fd, [track], active_track_ids=set())

    # --- Assert: Should re-identify as track 42 ---
    assert len(result.matches) == 1,                 "Should match 1 track"
    assert result.matches[0].matched_track_id == 42, "Should recover track #42"
    assert result.matches[0].is_confident is True
    sim = result.matches[0].similarity_score
    assert sim > 0.3, f"Similarity should be positive (same person), got {sim:.3f}"

    print(f"\n         Person #42 re-identified with similarity={sim:.3f}  ✓")


def test_get_stats():
    """Stats dict should have correct keys and types."""
    cfg  = make_config()
    from reid.reidentifier import ReIdentifier
    reid = ReIdentifier(cfg)
    reid.model = None

    track, base_emb = make_track(1, n_gallery=5)
    fd = make_frame_detections(1, n=2, embeddings=[base_emb, None])
    reid.match(fd, [track], set())

    stats = reid.get_stats()
    assert "total_queries"    in stats
    assert "total_matches"    in stats
    assert "total_rejections" in stats
    assert "match_rate"       in stats
    assert 0.0 <= stats["match_rate"] <= 1.0


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phantom Tracker — Re-ID Module Tests")
    print("=" * 60)

    print("\n[Feature Bank]")
    test("add and similarity",           test_feature_bank_add_and_similarity)
    test("max size enforcement",         test_feature_bank_max_size)
    test("best_match correct track",     test_feature_bank_best_match_correct_track)
    test("excludes claimed IDs",         test_feature_bank_excludes_claimed_ids)

    print("\n[ReIdentifier]")
    test("init without model",           test_reidentifier_init_no_model)
    test("match() no model → empty",     test_match_no_model_returns_empty)
    test("match() with precomputed emb", test_match_with_precomputed_embeddings)
    test("match() rejects low sim",      test_match_rejects_below_threshold)
    test("no double assignment",         test_match_no_double_assignment)
    test("stats dict",                   test_get_stats)

    print("\n[Midterm Demo Scenario]")
    test("person leaves → returns → re-identified", test_reid_scenario_person_leaves_and_returns)

    print(f"\n{'='*60}")
    print(f"  Results: {PASSED} passed, {FAILED} failed")
    print(f"{'='*60}\n")

    if FAILED > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
