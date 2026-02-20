"""FPS counter with rolling average."""
import time
from collections import deque


class FPSCounter:
    def __init__(self, window_size: int = 60):
        self.window = deque(maxlen=window_size)
        self.all_times = []
        self.last_time = time.time()

    def tick(self):
        now = time.time()
        dt = now - self.last_time
        self.window.append(dt)
        self.all_times.append(dt)
        self.last_time = now

    def get_fps(self) -> float:
        if not self.window:
            return 0.0
        avg_dt = sum(self.window) / len(self.window)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    def get_avg_fps(self) -> float:
        if not self.all_times:
            return 0.0
        avg_dt = sum(self.all_times) / len(self.all_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0
