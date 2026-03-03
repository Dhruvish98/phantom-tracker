"""
Phantom Tracker — Main Pipeline
Run:  python main.py                                       # webcam
      python main.py --input video.mp4                     # video file
      python main.py --input video.mp4 --classes person    # person-only
      python main.py --input video.mp4 --half              # FP16 on GPU
Keys: [q] quit  [t] trails  [g] ghost  [p] predicted  [i] IDs
      [f] FPS   [h] heatmap  [d] dashboard
"""

import cv2, time, argparse
import numpy as np
from core.interfaces import FrameState, PipelineConfig, AnalyticsSnapshot
from detection.detector import Detector
from tracking.tracker import Tracker
from reid.reidentifier import ReIdentifier
from visualization.visualizer import Visualizer
from utils.fps_counter import FPSCounter
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PhantomTracker:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.frame_id = 0
        logger.info("Initializing Phantom Tracker...")
        self.detector = Detector(config)
        self.tracker = Tracker(config)
        self.reidentifier = ReIdentifier(config)
        self.visualizer = Visualizer(config)
        self.fps = FPSCounter()
        logger.info("All modules ready.")

    def process_frame(self, frame: np.ndarray) -> FrameState:
        ts = time.time()
        self.frame_id += 1
        state = FrameState(frame_id=self.frame_id, timestamp=ts, raw_frame=frame)

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
                self.tracker.apply_reid_results(state.reid_results)
        # Stage 4: Visualize
        analytics = self.tracker.get_analytics(self.frame_id, ts)
        state.output_frame = self.visualizer.render(frame, state, analytics, self.fps.get_fps())
        self.fps.tick()
        return state

    def run(self, input_source=None, display=True, output_path=None):
        source = input_source if input_source is not None else self.config.input_source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Cannot open: {source}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        w, h = int(cap.get(3)), int(cap.get(4))
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        logger.info(f"Source: {w}x{h} @ {vfps:.0f}fps — Press q to quit")

        writer = None
        if output_path:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), vfps, (w, h))

        try:
            while True:
                ret, frame = cap.read()
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
            cap.release()
            if writer: writer.release()
            if display: cv2.destroyAllWindows()
        logger.info(f"Done. {self.frame_id} frames, avg {self.fps.get_avg_fps():.1f} FPS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--yolo-model", type=str, default="yolo11s.pt")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--classes", type=str, default=None,
                   help="Comma-separated class filter, e.g. 'person,backpack'")
    p.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    p.add_argument("--no-display", action="store_true")
    a = p.parse_args()
    detect_classes = [c.strip() for c in a.classes.split(",")] if a.classes else []
    cfg = PipelineConfig(yolo_model=a.yolo_model, yolo_confidence=a.confidence,
                         detect_classes=detect_classes, yolo_half=a.half,
                         input_source=a.input if a.input else 0)
    PhantomTracker(cfg).run(a.input if a.input else 0, not a.no_display, a.output)
