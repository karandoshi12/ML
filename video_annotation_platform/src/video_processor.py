"""Video ingestion and frame extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    path: str
    fps: float
    duration: float
    width: int
    height: int
    total_frames: int


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray  # BGR, shape (H, W, 3)


class VideoProcessor:
    """Load an MP4 (or other supported) video and yield frames."""

    def __init__(self, video_path: str, config: dict):
        self.video_path = os.path.abspath(video_path)
        self.config = config
        self._cap: cv2.VideoCapture | None = None

        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        ext = os.path.splitext(video_path)[1].lstrip(".").lower()
        supported = config.get("video", {}).get("supported_formats", ["mp4", "avi", "mov", "mkv"])
        if ext not in supported:
            raise ValueError(f"Unsupported format '.{ext}'. Supported: {supported}")

        self._open()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_metadata(self) -> VideoMetadata:
        cap = self._get_cap()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps if fps > 0 else 0.0
        return VideoMetadata(
            path=self.video_path,
            fps=fps,
            duration=duration,
            width=w,
            height=h,
            total_frames=total,
        )

    def extract_frames(self) -> Iterator[Frame]:
        """Yield every frame from the video, optionally resizing."""
        cap = self._get_cap()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        resize = self.config.get("video", {}).get("resize_frames", False)
        target = tuple(self.config.get("video", {}).get("target_size", [640, 480]))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = idx / fps
            if resize:
                frame = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
            yield Frame(index=idx, timestamp=ts, image=frame)
            idx += 1

    def extract_frame_at(self, index: int) -> Frame:
        """Extract a single frame by index."""
        cap = self._get_cap()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if not ret:
            raise IndexError(f"Could not read frame at index {index}")
        resize = self.config.get("video", {}).get("resize_frames", False)
        target = tuple(self.config.get("video", {}).get("target_size", [640, 480]))
        if resize:
            frame = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
        return Frame(index=index, timestamp=index / fps, image=frame)

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _open(self):
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {self.video_path}")

    def _get_cap(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            self._open()
        return self._cap

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
