"""
Grounding DINO Detector
=======================
Open-vocabulary object detection via HuggingFace's `IDEA-Research/grounding-dino-tiny`.

Why open-vocabulary?
  YOLO is constrained to its 80 COCO classes. Grounding DINO accepts an
  arbitrary natural-language prompt and finds objects matching that prompt:

    "person with red backpack"
    "abandoned luggage"
    "dog on a leash"

Trade-off vs YOLO:
  - SLOWER: ~5-10 FPS on RTX 2050 vs YOLO's 70+ FPS
  - HEAVIER: ~150M params (tiny) vs YOLO11s 9.4M
  - BUT enables tracking targets that YOLO can't recognize at all, and lets
    a user define what to track at runtime via prompt instead of model retrain

Output is the same `Detection` / `FrameDetections` schema YOLO produces, so
the rest of the pipeline (tracking, Re-ID, visualization) is unaffected.

Prompt syntax (Grounding DINO convention):
  Multiple categories are joined with periods, e.g. "person. backpack. dog."
  Each "category" can itself be a phrase ("person with red backpack").
"""
from __future__ import annotations

import time

import numpy as np

from core.interfaces import Detection, FrameDetections, PipelineConfig
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Default checkpoint - "tiny" gives an acceptable speed/accuracy trade-off on
# our hardware. Switch to "grounding-dino-base" for higher accuracy at ~3x
# slower inference (and ~340M params).
DEFAULT_CHECKPOINT = "IDEA-Research/grounding-dino-tiny"


class GroundingDinoDetector:
    """
    Drop-in alternative to detection.detector.Detector that uses Grounding DINO
    instead of YOLO. Same .detect() signature, same FrameDetections output.
    """

    def __init__(
        self,
        config: PipelineConfig,
        camera_id: str = "default",
        prompt: str = "person.",
        checkpoint: str = DEFAULT_CHECKPOINT,
    ):
        self.config = config
        self.camera_id = camera_id
        self.prompt = self._normalize_prompt(prompt)
        self.checkpoint = checkpoint
        self.processor = None
        self.model = None
        self.device = "cpu"
        self._load_model()

    # ── lifecycle ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        """Grounding DINO expects period-separated category phrases. Be lenient."""
        prompt = prompt.strip().lower()
        if not prompt:
            return "person."
        # Add trailing period if user forgot.
        if not prompt.endswith("."):
            prompt = prompt + "."
        return prompt

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as e:
            logger.error(
                f"Grounding DINO requires `transformers` and `torch`: {e}. "
                "Install with: pip install transformers torch"
            )
            raise

        self.device = (
            "cuda:0"
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else "cpu"
        )
        logger.info(
            f"[GroundingDINO] Loading {self.checkpoint} on {self.device} "
            f"(prompt={self.prompt!r}). First load downloads ~150MB; cached after."
        )
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.checkpoint
        ).to(self.device)
        self.model.eval()
        logger.info("[GroundingDINO] model ready")

    # ── inference ──────────────────────────────────────────────────────

    def detect(
        self, frame: np.ndarray, frame_id: int, timestamp: float
    ) -> FrameDetections:
        """Run detection on one frame; return FrameDetections in the standard schema."""
        start = time.time()
        detections = self._infer(frame)
        elapsed_ms = (time.time() - start) * 1000
        return FrameDetections(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            source="grounding_dino",
            inference_time_ms=elapsed_ms,
            camera_id=self.camera_id,
        )

    def _infer(self, frame: np.ndarray) -> list[Detection]:
        """Core Grounding DINO inference. Converts BGR frame -> Detection list."""
        import torch
        from PIL import Image

        # transformers.AutoProcessor expects PIL RGB images.
        rgb = frame[:, :, ::-1]  # BGR -> RGB
        image = Image.fromarray(rgb)

        inputs = self.processor(
            images=image, text=self.prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # post_process_grounded_object_detection rescales boxes back to image size
        # and applies the prompt-token similarity threshold.
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        # transformers >= 5 uses `text_threshold` / `threshold` keyword names;
        # we pass both spellings for forward/backward compatibility.
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.config.yolo_confidence,
                text_threshold=0.25,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            # Newer transformers signature
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                threshold=self.config.yolo_confidence,
                text_threshold=0.25,
                target_sizes=target_sizes,
            )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]  # list of strings (the matched phrase per box)

        detections: list[Detection] = []
        for bbox, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = bbox.astype(np.float32)
            cls_name = str(label).strip().rstrip(".") or "object"
            detections.append(
                Detection(
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                    confidence=float(score),
                    class_name=cls_name,
                    class_id=-1,  # open-vocab: no fixed class id
                )
            )
        return detections

    # ── introspection ──────────────────────────────────────────────────

    def benchmark(self, frame: np.ndarray, n: int = 30) -> dict:
        """Run n inferences and return mean/std latency. Used for FPS profiling."""
        times = []
        for _ in range(n):
            t0 = time.time()
            self._infer(frame)
            times.append((time.time() - t0) * 1000)
        return {
            "model": self.checkpoint,
            "prompt": self.prompt,
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "fps": 1000.0 / float(np.mean(times)),
        }
