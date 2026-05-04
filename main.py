"""
Phantom Tracker — Main Pipeline
Single-camera:
    python main.py                                              # webcam
    python main.py --input video.mp4                            # video file
    python main.py --input video.mp4 --classes person           # person-only
    python main.py --input video.mp4 --half                     # FP16 on GPU

Open-vocabulary detection (Grounding DINO):
    python main.py --input video.mp4 --prompt "person with red backpack"
    python main.py --input video.mp4 --prompt "dog. person. bicycle."

Multi-camera (cross-camera Re-ID, non-overlapping FOVs):
    python main.py --cameras "A=0,B=http://192.168.29.163:8080/video"
    python main.py --cameras "A=demos/slow_walkers.mp4,B=demos/dog_park.mp4"

Keys: [q] quit  [t] trails  [g] ghost  [p] predicted  [i] IDs
      [f] FPS   [h] heatmap  [d] dashboard
"""

import cv2, time, argparse
import numpy as np
from core.interfaces import FrameState, PipelineConfig
from detection.detector import Detector
from detection.grounding_dino import GroundingDinoDetector
from tracking.tracker import Tracker
from reid.reidentifier import ReIdentifier
from reid.cross_camera import CrossCameraCoordinator
from visualization.visualizer import Visualizer
from utils.camera import Camera
from utils.fps_counter import FPSCounter
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PhantomTracker:
    def __init__(self, config: PipelineConfig, camera_id: str = "default"):
        self.config = config
        self.camera_id = camera_id
        self.frame_id = 0
        logger.info(f"Initializing Phantom Tracker (camera_id={camera_id})...")
        # Detector dispatch: open-vocabulary Grounding DINO when configured,
        # otherwise YOLO. Both implement the same .detect() -> FrameDetections.
        if config.use_grounding_dino:
            logger.info(
                f"Using Grounding DINO detector with prompt={config.grounding_dino_prompt!r}"
            )
            self.detector = GroundingDinoDetector(
                config, camera_id=camera_id,
                prompt=config.grounding_dino_prompt,
                checkpoint=config.grounding_dino_checkpoint,
            )
        else:
            self.detector = Detector(config, camera_id=camera_id)
        self.tracker = Tracker(config, camera_id=camera_id)
        self.reidentifier = ReIdentifier(config)
        self.visualizer = Visualizer(config)
        self.fps = FPSCounter()
        logger.info("All modules ready.")

    def process_frame(self, frame: np.ndarray) -> FrameState:
        ts = time.time()
        self.frame_id += 1
        state = FrameState(frame_id=self.frame_id, timestamp=ts,
                           raw_frame=frame, camera_id=self.camera_id)

        # Stage 1: Detect
        state.detections = self.detector.detect(frame, self.frame_id, ts)
        # Stage 2: Track
        state.active_tracks, state.occluded_tracks, state.lost_tracks = \
            self.tracker.update(state.detections, frame)
        # Stage 2.5: Update appearance gallery for active tracks (feeds Re-ID)
        for track in state.active_tracks:
            if self.frame_id % 5 == 0:  # every 5th frame to save compute
                self.reidentifier.update_gallery(track, frame)

        # Stage 3: Re-ID
        if state.lost_tracks and state.detections:
            state.reid_results = self.reidentifier.match(
                state.detections, state.lost_tracks,
                {t.track_id for t in state.active_tracks},
                frame=frame)
            if state.reid_results and state.reid_results.matches:
                self.tracker.apply_reid_results(state.reid_results, state.detections)
        # Stage 4: Visualize
        analytics = self.tracker.get_analytics(self.frame_id, ts)
        state.output_frame = self.visualizer.render(frame, state, analytics, self.fps.get_fps())
        self.fps.tick()
        return state

    def run(self, input_source=None, display=True, output_path=None):
        source = input_source if input_source is not None else self.config.input_source
        try:
            camera = Camera(
                source=source,
                camera_id=self.camera_id,
                target_width=self.config.frame_width,
                target_height=self.config.frame_height,
            )
        except RuntimeError as e:
            logger.error(str(e))
            return
        logger.info(f"Source ready ({camera.kind}) - Press q to quit")

        writer = None
        if output_path:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     camera.fps, (camera.width, camera.height))

        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    break
                state = self.process_frame(frame)
                if writer and state.output_frame is not None:
                    writer.write(state.output_frame)
                if display and state.output_frame is not None:
                    cv2.imshow("Phantom Tracker", state.output_frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"): break
                    # Toggle visualization features
                    toggles = [
                        ("t", "show_trails"),
                        ("g", "show_ghost_outlines"),
                        ("p", "show_predicted_path"),
                        ("i", "show_ids"),
                        ("f", "show_fps")
                    ]
                    for key, attr in toggles:
                        if k == ord(key):
                            setattr(self.config, attr, not getattr(self.config, attr))
                            logger.info(f"{attr}: {getattr(self.config, attr)}")

                    # Toggle visualizer-specific features
                    if k == ord("h"):
                        self.visualizer.show_heatmap = not self.visualizer.show_heatmap
                        logger.info(f"Heatmap: {self.visualizer.show_heatmap}")
                    if k == ord("d"):
                        self.visualizer.show_dashboard = not self.visualizer.show_dashboard
                        logger.info(f"Dashboard: {self.visualizer.show_dashboard}")
        finally:
            camera.release()
            if writer: writer.release()
            if display: cv2.destroyAllWindows()
        logger.info(f"Done. {self.frame_id} frames, avg {self.fps.get_avg_fps():.1f} FPS")


def _draw_text_with_outline(img, text, org, font_scale=0.7, color=(255, 255, 255),
                             thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Draw text with a black outline so it reads on any background."""
    cv2.putText(img, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_camera_panel(frame, camera_id, active_tracks, header_h=36):
    """Add a header strip above the frame showing camera id, count, and visible global ids."""
    if frame is None:
        return None
    panel = np.zeros((frame.shape[0] + header_h, frame.shape[1], 3), dtype=np.uint8)
    # Header bar
    panel[:header_h] = (32, 32, 32)
    # Camera label
    _draw_text_with_outline(panel, f"CAM: {camera_id}", (10, 26), font_scale=0.75)
    # Active count
    count_text = f"active: {len(active_tracks)}"
    _draw_text_with_outline(panel, count_text, (170, 26),
                             font_scale=0.6, color=(180, 220, 255))
    # Visible global ids (as a compact list)
    gids = sorted({t.global_id for t in active_tracks if t.global_id is not None})
    if gids:
        gid_text = "G: " + ",".join(str(g) for g in gids[:8])
        if len(gids) > 8:
            gid_text += f"+{len(gids) - 8}"
        _draw_text_with_outline(panel, gid_text, (320, 26),
                                 font_scale=0.6, color=(150, 255, 150))
    # Place the actual frame below the header
    panel[header_h:] = frame
    return panel


def _draw_handoff_banner(stitched, recent_matches, panel_widths, target_height,
                         header_h=36, banner_h=44):
    """Draw a banner at the bottom listing recent cross-camera handoffs."""
    if not recent_matches:
        return stitched
    h, w = stitched.shape[:2]
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (0, 80, 0)  # dark green background to suggest a successful handoff
    cv2.line(banner, (0, 0), (w, 0), (0, 200, 0), 2)
    # Show up to 2 most recent matches, freshest first
    for i, m in enumerate(recent_matches[:2]):
        text = (f"HANDOFF: G{m['global_id']} "
                f"{m['from_cam']} -> {m['to_cam']} "
                f"(sim {m['similarity']:.2f})")
        y = 18 + i * 20
        _draw_text_with_outline(banner, text, (12, y),
                                 font_scale=0.55, color=(180, 255, 180))
    return np.vstack([stitched, banner])


def _draw_footer_stats(stitched, stats):
    """Append a one-line footer summarizing cross-camera totals."""
    h, w = stitched.shape[:2]
    footer = np.zeros((24, w, 3), dtype=np.uint8)
    footer[:] = (24, 24, 24)
    text = (f"Cross-cam matches: {stats['cross_camera_matches']}  |  "
            f"Total identities: {stats['total_global_identities']}  |  "
            f"Deferred (no embedding yet): {stats['deferred_no_embedding']}")
    _draw_text_with_outline(footer, text, (10, 17), font_scale=0.5,
                             color=(200, 200, 200))
    return np.vstack([stitched, footer])


def _stitch_horizontal(frames: list, camera_ids: list,
                       all_active_tracks: list,
                       recent_matches: list = None,
                       coordinator_stats: dict = None,
                       target_height: int = 540) -> np.ndarray:
    """
    Compose a multi-camera display:
      [ panel header | camera frame ]  [ panel header | camera frame ]
      [ HANDOFF banner (green, fades) ]
      [ cross-camera stats footer ]
    """
    panels = []
    for frame, cam_id, active in zip(frames, camera_ids, all_active_tracks):
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = target_height / h
        resized = cv2.resize(frame, (int(w * scale), target_height))
        panels.append(_draw_camera_panel(resized, cam_id, active))

    if not panels:
        return np.zeros((target_height, 100, 3), dtype=np.uint8)
    stitched = np.hstack(panels)

    if recent_matches:
        stitched = _draw_handoff_banner(
            stitched, recent_matches,
            [p.shape[1] for p in panels], target_height,
        )
    if coordinator_stats:
        stitched = _draw_footer_stats(stitched, coordinator_stats)
    return stitched


def _parse_cameras_arg(arg: str) -> list[tuple[str, str]]:
    """Parse '--cameras A=src1,B=src2' into [('A', 'src1'), ('B', 'src2')]."""
    pairs = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Bad --cameras spec '{chunk}'; expected NAME=SOURCE")
        name, source = chunk.split("=", 1)
        pairs.append((name.strip(), source.strip()))
    if not pairs:
        raise ValueError("--cameras must specify at least one NAME=SOURCE pair")
    return pairs


class MultiCameraPipeline:
    """
    Orchestrates N PhantomTracker instances (one per camera) and a shared
    CrossCameraCoordinator. Reads one frame from each camera per tick,
    processes them sequentially through their per-camera pipelines, runs the
    cross-camera coordinator to bind tracks to global identities, and renders
    a side-by-side display.
    """

    def __init__(self, config: PipelineConfig, camera_specs: list[tuple[str, str]]):
        self.config = config
        self.camera_specs = camera_specs  # [(camera_id, source), ...]
        logger.info(f"Initializing MultiCameraPipeline with {len(camera_specs)} cameras")
        self.pipelines: dict[str, PhantomTracker] = {
            cam_id: PhantomTracker(config, camera_id=cam_id)
            for cam_id, _src in camera_specs
        }
        self.coordinator = CrossCameraCoordinator(config)

    def run(self, display: bool = True, output_path: str = None) -> None:
        cameras: dict[str, Camera] = {}
        try:
            for cam_id, source in self.camera_specs:
                cameras[cam_id] = Camera(
                    source=source, camera_id=cam_id,
                    target_width=self.config.frame_width,
                    target_height=self.config.frame_height,
                )
        except RuntimeError as e:
            logger.error(str(e))
            for c in cameras.values():
                c.release()
            return

        writer = None  # left as a future enhancement for multi-cam recording

        try:
            while True:
                # Step 1: read one frame from each camera (sequential)
                frames: dict[str, np.ndarray] = {}
                any_dropped = False
                for cam_id, cam in cameras.items():
                    ret, frame = cam.read()
                    if not ret:
                        logger.info(f"[{cam_id}] stream ended")
                        any_dropped = True
                        break
                    frames[cam_id] = frame
                if any_dropped:
                    break

                # Step 2: per-camera pipeline (detection -> track -> Re-ID -> render)
                states: dict[str, FrameState] = {}
                for cam_id, frame in frames.items():
                    states[cam_id] = self.pipelines[cam_id].process_frame(frame)

                # Step 3: cross-camera coordination (assign global_ids)
                tracks_by_camera = {
                    cam_id: state.active_tracks for cam_id, state in states.items()
                }
                self.coordinator.register_or_match(tracks_by_camera, frames)

                # Step 4: side-by-side display
                if display:
                    output_frames = [states[cid].output_frame for cid, _ in self.camera_specs]
                    cam_ids = [cid for cid, _ in self.camera_specs]
                    active_tracks_per_cam = [states[cid].active_tracks for cid, _ in self.camera_specs]
                    stitched = _stitch_horizontal(
                        output_frames, cam_ids, active_tracks_per_cam,
                        recent_matches=self.coordinator.get_recent_matches(within_s=3.0),
                        coordinator_stats=self.coordinator.stats(),
                    )
                    cv2.imshow("Phantom Tracker — Multi-Camera", stitched)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    # Visualization toggles broadcast to every per-camera pipeline
                    cfg_toggles = [
                        ("t", "show_trails"), ("g", "show_ghost_outlines"),
                        ("p", "show_predicted_path"), ("i", "show_ids"),
                        ("f", "show_fps"),
                    ]
                    for key, attr in cfg_toggles:
                        if k == ord(key):
                            new_val = not getattr(self.config, attr)
                            setattr(self.config, attr, new_val)
                            logger.info(f"{attr}: {new_val}")
                    if k == ord("h"):
                        for pipe in self.pipelines.values():
                            pipe.visualizer.show_heatmap = not pipe.visualizer.show_heatmap
                    if k == ord("d"):
                        for pipe in self.pipelines.values():
                            pipe.visualizer.show_dashboard = not pipe.visualizer.show_dashboard
        finally:
            for c in cameras.values():
                c.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        logger.info(f"Done. Cross-camera stats: {self.coordinator.stats()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None,
                   help="Single-camera input source (file, URL, or webcam index)")
    p.add_argument("--cameras", type=str, default=None,
                   help='Multi-camera spec, e.g. "A=0,B=http://192.168.29.163:8080/video"')
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--yolo-model", type=str, default="yolo11s.pt")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--classes", type=str, default=None,
                   help="Comma-separated class filter, e.g. 'person,backpack'")
    p.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    p.add_argument("--prompt", type=str, default=None,
                   help='Open-vocabulary prompt for Grounding DINO, e.g. '
                        '"person with red backpack". Switches detector from YOLO to '
                        'Grounding DINO; --classes is ignored when set.')
    p.add_argument("--grounding-dino-checkpoint", type=str,
                   default="IDEA-Research/grounding-dino-tiny",
                   help="HuggingFace checkpoint for Grounding DINO (default: tiny)")
    p.add_argument("--no-display", action="store_true")
    a = p.parse_args()
    detect_classes = [c.strip() for c in a.classes.split(",")] if a.classes else []

    cfg = PipelineConfig(
        yolo_model=a.yolo_model, yolo_confidence=a.confidence,
        detect_classes=detect_classes, yolo_half=a.half,
        input_source=a.input if a.input else 0,
        use_grounding_dino=bool(a.prompt),
        grounding_dino_prompt=a.prompt or "person.",
        grounding_dino_checkpoint=a.grounding_dino_checkpoint,
    )

    if a.cameras:
        # Multi-camera mode
        if a.input:
            logger.warning("--input is ignored when --cameras is set")
        camera_specs = _parse_cameras_arg(a.cameras)
        MultiCameraPipeline(cfg, camera_specs).run(
            display=not a.no_display, output_path=a.output
        )
    else:
        # Single-camera mode (existing behavior)
        PhantomTracker(cfg).run(a.input if a.input else 0, not a.no_display, a.output)
