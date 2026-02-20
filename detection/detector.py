"""
Detection Module — Owner: DIVYANSH
====================================
YOLOv11 (primary, every frame) + Grounding DINO (on-demand, post-MVP).

MVP target: YOLO detection on live webcam at 30+ FPS.

TODO (Divyansh):
  [ ] Test yolo11n vs yolo11s vs yolo11m — find the speed/accuracy sweet spot
  [ ] Add TensorRT export for faster inference (post-MVP)
  [ ] Implement Grounding DINO loader and inference (post-MVP)
  [ ] Add class filtering (e.g., only detect "person" class for demos)
"""

import numpy as np
import time
from core.interfaces import Detection, FrameDetections, PipelineConfig
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Detector:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.yolo_model = None
        self._load_yolo()

    def _load_yolo(self):
        """Load YOLOv11. Requires: pip install ultralytics"""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.config.yolo_model)
            logger.info(f"YOLO loaded: {self.config.yolo_model}")
        except ImportError:
            logger.error("Install ultralytics: pip install ultralytics")
            raise

    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> FrameDetections:
        """Run detection on one frame. Called every frame by main.py."""
        start = time.time()
        detections = self._detect_yolo(frame)
        elapsed = (time.time() - start) * 1000
        return FrameDetections(
            frame_id=frame_id, timestamp=timestamp,
            detections=detections, source="yolo", inference_time_ms=elapsed)

    def _detect_yolo(self, frame: np.ndarray) -> list[Detection]:
        """
        Core YOLO inference. Converts ultralytics output → our Detection format.

        ultralytics result.boxes has:
          .xyxy  — bounding boxes [x1,y1,x2,y2]
          .conf  — confidence scores
          .cls   — class indices
        """
        results = self.yolo_model(frame, conf=self.config.yolo_confidence, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                bbox = result.boxes.xyxy[i].cpu().numpy().astype(np.float32)
                conf = float(result.boxes.conf[i].cpu().numpy())
                cls_id = int(result.boxes.cls[i].cpu().numpy())
                detections.append(Detection(
                    bbox=bbox, confidence=conf,
                    class_name=self.yolo_model.names[cls_id], class_id=cls_id))
        return detections

    def _detect_grounding_dino(self, frame: np.ndarray, prompt: str) -> list[Detection]:
        """
        TODO (Divyansh — post-MVP): Grounding DINO open-vocabulary detection.

        Approach options:
          1. groundingdino from IDEA-Research (pip install groundingdino)
          2. HuggingFace transformers AutoModelForZeroShotObjectDetection
          3. autodistill-grounding-dino wrapper

        Key insight: This only needs to run ONCE to init a track.
        After that, the tracker maintains it. So 8-12 FPS is fine.
        """
        logger.warning("Grounding DINO not yet implemented")
        return []

    def benchmark(self, frame: np.ndarray, n=100) -> dict:
        """Run n inferences, report timing stats. Use for midterm FPS numbers."""
        times = []
        for _ in range(n):
            t0 = time.time()
            self._detect_yolo(frame)
            times.append((time.time() - t0) * 1000)
        return {"model": self.config.yolo_model, "mean_ms": np.mean(times),
                "fps": 1000 / np.mean(times), "std_ms": np.std(times)}
