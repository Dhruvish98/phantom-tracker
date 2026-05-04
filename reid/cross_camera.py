"""
Cross-Camera Re-ID Coordinator
==============================
Maintains globally consistent identities across multiple camera feeds
(non-overlapping FOVs assumed).

Per-camera Re-ID (reid/reidentifier.py) handles within-camera occlusion.
This module handles between-camera handoff: person leaves camera A's FOV,
transits invisibly for a few seconds, then appears in camera B's FOV. We
bind both per-camera tracks to the same global_id so the visualization and
analytics treat them as the same person.

Algorithm (called once per pipeline tick from MultiCameraPipeline):

  For every active per-camera track:
    a) If the track is already bound to a global identity:
         - touch the identity (update last_seen_ts, last_camera_id)
         - append the current embedding to the identity's gallery
    b) Else (a fresh track with no global_id):
         - read its current embedding (track.last_embedding, populated by
           per-camera Re-ID's gallery updates)
         - if embedding is None yet, defer matching to a later frame
         - filter global identities by transit-time prior:
             keep identities last-seen on a *different* camera, within
             [min_transit_s, max_transit_s] ago
         - compute weighted cosine similarity to each candidate's gallery
         - if best similarity > cross_camera_threshold: bind to that global_id
         - otherwise register a brand-new global_id

Identity color is updated on bind so the visualizer renders the same person
with the same color across cameras.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.interfaces import Track, PipelineConfig
from utils.colors import generate_unique_color
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class GlobalIdentity:
    """Cross-camera canonical identity."""
    global_id: int
    embeddings: list = field(default_factory=list)   # [(emb, ts), ...]
    last_camera_id: str = ""
    last_seen_ts: float = 0.0
    first_seen_ts: float = 0.0
    color: tuple = (255, 255, 255)
    # Map (camera_id -> per-camera track_id) for tracks bound to this identity
    bindings: dict = field(default_factory=dict)


class CrossCameraCoordinator:
    """
    Coordinates global identities across per-camera trackers.

    Designed for non-overlapping FOVs but tolerates brief overlap and
    same-camera re-association. Identities currently bound to an active
    track on a camera are excluded from re-binding (no double-assignment),
    but identities last seen on the same camera that are no longer actively
    bound remain candidates - this lets cross-cam act as a backstop when
    per-camera Re-ID misses a re-appearance.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.identities: dict[int, GlobalIdentity] = {}
        self._next_global_id = 1
        self.threshold = config.cross_camera_threshold
        self.min_transit_s = config.cross_camera_min_transit_s
        self.max_transit_s = config.cross_camera_max_transit_s
        self.gallery_size = config.cross_camera_gallery_size

        # Diagnostics
        self.match_count = 0
        self.new_identity_count = 0
        self.deferred_count = 0   # tracks skipped because no embedding yet

        # Recent handoff events for the visualizer to render banners.
        # Each entry: {"ts": float, "global_id": int, "from_cam": str,
        #              "to_cam": str, "similarity": float}
        self.recent_matches: list[dict] = []

    # ── public API ─────────────────────────────────────────────────────

    def register_or_match(self, tracks_by_camera: dict[str, list[Track]],
                          frame_by_camera: Optional[dict] = None) -> None:
        """Run cross-camera matching for one pipeline tick."""
        now = time.time()

        # Snapshot which global_ids are currently bound on each camera so we
        # don't bind two simultaneously-visible tracks to the same identity.
        currently_bound_per_camera: dict[str, set] = {
            cam_id: {t.global_id for t in tracks if t.global_id is not None}
            for cam_id, tracks in tracks_by_camera.items()
        }

        for camera_id, tracks in tracks_by_camera.items():
            for track in tracks:
                if track.global_id is not None:
                    self._touch_identity(track, camera_id, now)
                    continue

                emb = track.last_embedding
                if emb is None:
                    # Per-camera Re-ID hasn't extracted an embedding yet.
                    # Wait for next frame.
                    self.deferred_count += 1
                    continue

                gid = self._find_match(emb, camera_id,
                                       exclude_gids=currently_bound_per_camera[camera_id],
                                       now=now)
                if gid is not None:
                    self._bind(track, camera_id, gid, emb, now)
                    currently_bound_per_camera[camera_id].add(gid)
                else:
                    new_gid = self._allocate_identity(track, camera_id, emb, now)
                    self._bind(track, camera_id, new_gid, emb, now)
                    currently_bound_per_camera[camera_id].add(new_gid)

    def stats(self) -> dict:
        return {
            "total_global_identities": len(self.identities),
            "next_global_id": self._next_global_id,
            "cross_camera_matches": self.match_count,
            "new_identities": self.new_identity_count,
            "deferred_no_embedding": self.deferred_count,
        }

    def get_recent_matches(self, within_s: float = 3.0) -> list[dict]:
        """Return handoff events within the last `within_s` seconds, freshest first.
        Old entries are pruned as a side effect to keep memory bounded."""
        now = time.time()
        cutoff = now - within_s
        # Prune old (anything older than 60s — well past any reasonable display window)
        prune_cutoff = now - 60.0
        self.recent_matches = [m for m in self.recent_matches if m["ts"] >= prune_cutoff]
        return [m for m in reversed(self.recent_matches) if m["ts"] >= cutoff]

    # ── matching ───────────────────────────────────────────────────────

    def _find_match(self, query_emb: np.ndarray, camera_id: str,
                    exclude_gids: set, now: float) -> Optional[int]:
        """Return the best matching global_id, or None if no candidate clears the threshold."""
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        best_gid = None
        best_score = -1.0
        candidates_considered = 0

        for gid, ident in self.identities.items():
            # exclude_gids already lists identities currently bound to an active
            # track on *this* camera — never re-bind a second track to the same
            # identity simultaneously.
            if gid in exclude_gids:
                continue
            # Transit-time prior: identity must have been seen recently enough
            # to plausibly be the same person (within max_transit_s). Note we
            # *do* allow same-camera matches here: if the per-camera Re-ID
            # missed a re-appearance, cross-cam acts as a backstop. The
            # exclude_gids check above already prevents binding two
            # simultaneously-visible tracks to the same identity.
            elapsed = now - ident.last_seen_ts
            if elapsed < self.min_transit_s or elapsed > self.max_transit_s:
                continue
            if not ident.embeddings:
                continue

            candidates_considered += 1
            score = self._gallery_similarity(q, ident.embeddings)
            if score > best_score:
                best_score = score
                best_gid = gid

        if best_gid is not None and best_score >= self.threshold:
            logger.info(
                f"[CrossCam] MATCH: {camera_id} -> global_id={best_gid} "
                f"sim={best_score:.3f} (threshold={self.threshold}, "
                f"candidates_considered={candidates_considered})"
            )
            self.match_count += 1
            # Record handoff for the visualizer; the source camera is whatever
            # the identity was last seen on (a *different* camera by construction).
            from_cam = self.identities[best_gid].last_camera_id
            self.recent_matches.append({
                "ts": now,
                "global_id": best_gid,
                "from_cam": from_cam,
                "to_cam": camera_id,
                "similarity": best_score,
            })
            return best_gid

        # Diagnostic: log near-misses and 'no candidates' cases so we can tune.
        if candidates_considered == 0:
            logger.info(
                f"[CrossCam] no candidates: {camera_id} new track has no qualifying "
                f"global identities (transit window {self.min_transit_s}-{self.max_transit_s}s, "
                f"not currently bound on {camera_id}). "
                f"Total identities tracked: {len(self.identities)}"
            )
        else:
            logger.info(
                f"[CrossCam] near-miss: {camera_id} -> best global_id={best_gid} "
                f"sim={best_score:.3f} BELOW threshold={self.threshold} "
                f"({candidates_considered} candidates considered)"
            )
        return None

    @staticmethod
    def _gallery_similarity(query_q: np.ndarray, gallery: list) -> float:
        """Mean cosine similarity of a normalized query against a gallery of (emb, ts) pairs."""
        sims = []
        for emb, _ts in gallery:
            e = emb / (np.linalg.norm(emb) + 1e-8)
            sims.append(float(np.dot(query_q, e)))
        return float(np.mean(sims)) if sims else -1.0

    # ── bookkeeping ────────────────────────────────────────────────────

    def _allocate_identity(self, track: Track, camera_id: str,
                           emb: np.ndarray, now: float) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        self.identities[gid] = GlobalIdentity(
            global_id=gid,
            color=generate_unique_color(gid),
            last_camera_id=camera_id,
            last_seen_ts=now,
            first_seen_ts=now,
        )
        self.new_identity_count += 1
        logger.debug(f"[CrossCam] new global_id={gid} on {camera_id} (track #{track.track_id})")
        return gid

    def _bind(self, track: Track, camera_id: str, gid: int,
              emb: np.ndarray, now: float) -> None:
        """Bind a per-camera track to a global identity and propagate metadata."""
        ident = self.identities[gid]
        track.global_id = gid
        track.color = ident.color  # same person -> same color across cameras
        ident.bindings[camera_id] = track.track_id
        ident.last_camera_id = camera_id
        ident.last_seen_ts = now
        ident.embeddings.append((emb, now))
        if len(ident.embeddings) > self.gallery_size:
            ident.embeddings.pop(0)

    def _touch_identity(self, track: Track, camera_id: str, now: float) -> None:
        """Update the bound identity's last-seen state and gallery."""
        ident = self.identities.get(track.global_id)
        if ident is None:
            # Defensive: track points at a global_id we don't know about.
            # Re-allocate one rather than crash.
            if track.last_embedding is not None:
                track.global_id = self._allocate_identity(
                    track, camera_id, track.last_embedding, now
                )
            return
        ident.bindings[camera_id] = track.track_id
        ident.last_camera_id = camera_id
        ident.last_seen_ts = now
        # Append a new embedding sample if available and distinct from the last one.
        if track.last_embedding is not None:
            if not ident.embeddings or ident.embeddings[-1][0] is not track.last_embedding:
                ident.embeddings.append((track.last_embedding, now))
                if len(ident.embeddings) > self.gallery_size:
                    ident.embeddings.pop(0)
