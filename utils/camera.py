"""
Camera abstraction for Phantom Tracker.

Unifies USB webcams, video files, and network streams (RTSP/HTTP/MJPEG) under a
single interface with automatic reconnection on transient network failures.

Used by main.py and (eventually) the multi-camera orchestrator.
"""
import time
from typing import Optional, Union

import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


# Source strings that trigger network-camera handling (auto-reconnect on failure).
_NETWORK_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://", "mjpeg://")


def _normalize_source(source: Union[int, str]) -> Union[int, str]:
    """Convert a string like '0' to int 0 (USB device index)."""
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def _classify_source(source: Union[int, str]) -> str:
    """Return one of: 'usb', 'file', 'network'."""
    source = _normalize_source(source)
    if isinstance(source, int):
        return "usb"
    if isinstance(source, str) and source.lower().startswith(_NETWORK_PREFIXES):
        return "network"
    return "file"


class Camera:
    """
    Uniform camera interface over OpenCV VideoCapture.

    Handles three source kinds:
      - USB webcam by index (int): Camera(0, ...)
      - Video file by path:        Camera("demos/test.mp4", ...)
      - Network stream by URL:     Camera("http://192.168.1.42:8080/video", ...)

    Network sources auto-reconnect on transient failures (configurable). USB and
    file sources do not reconnect — a USB read failure means hardware is gone, a
    file read failure means EOF.

    Typical use:
        with Camera(source, camera_id="cam_A") as cam:
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                process(frame)
    """

    def __init__(
        self,
        source: Union[int, str],
        camera_id: str = "default",
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        max_reconnect_attempts: int = 10,
        reconnect_delay_s: float = 0.5,
    ):
        self.source = _normalize_source(source)
        self.camera_id = camera_id
        self.kind = _classify_source(self.source)
        self.target_width = target_width
        self.target_height = target_height
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s

        self.cap: Optional[cv2.VideoCapture] = None
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 30.0
        self._reconnect_count = 0
        self._open()

    # ── lifecycle ──────────────────────────────────────────────────────

    def _open(self) -> None:
        """Open the underlying VideoCapture and read source metadata."""
        logger.info(f"[Camera {self.camera_id}] Opening {self.kind} source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"[Camera {self.camera_id}] Cannot open source: {self.source}"
            )

        # USB webcams accept resolution requests; network/file sources usually ignore them.
        if self.target_width and self.target_height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        logger.info(
            f"[Camera {self.camera_id}] Opened: {self.width}x{self.height} @ "
            f"{self.fps:.0f}fps"
        )

    def release(self) -> None:
        """Release the underlying capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ── reading ────────────────────────────────────────────────────────

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.

        Returns:
            (True, frame) on success
            (False, None) on EOF (file source) or after exhausting reconnects (network)
        """
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self._reconnect_count = 0  # successful read resets the counter
            return True, frame

        # Read failed. Decide based on source kind.
        if self.kind == "file":
            return False, None  # EOF — don't reconnect
        if self.kind == "usb":
            logger.warning(f"[Camera {self.camera_id}] USB read failed (device unplugged?)")
            return False, None  # USB devices don't recover from a clean disconnect

        # Network source — try to reconnect.
        return self._try_reconnect()

    def _try_reconnect(self) -> tuple[bool, Optional[np.ndarray]]:
        """Attempt to reconnect a dropped network stream."""
        for attempt in range(1, self.max_reconnect_attempts + 1):
            self._reconnect_count += 1
            logger.warning(
                f"[Camera {self.camera_id}] Network stream dropped, "
                f"reconnect attempt {attempt}/{self.max_reconnect_attempts}"
            )
            self.release()
            time.sleep(self.reconnect_delay_s)
            try:
                self.cap = cv2.VideoCapture(self.source)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        logger.info(
                            f"[Camera {self.camera_id}] Reconnected after "
                            f"{attempt} attempt(s)"
                        )
                        return True, frame
            except Exception as e:
                logger.warning(f"[Camera {self.camera_id}] Reconnect error: {e}")

        logger.error(
            f"[Camera {self.camera_id}] Reconnect failed after "
            f"{self.max_reconnect_attempts} attempts; giving up"
        )
        return False, None

    # ── introspection ──────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    @property
    def reconnect_count(self) -> int:
        """Total reconnect attempts since last successful read (for diagnostics)."""
        return self._reconnect_count
