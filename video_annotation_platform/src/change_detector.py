"""Key-frame selection via MSE pixel diff + Farneback dense optical flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from .video_processor import Frame


@dataclass
class KeyFrame:
    frame: Frame
    change_score: float        # 0.0 – 1.0 normalised combined score
    is_scene_cut: bool = False  # True when MSE crosses hard-cut threshold


class ChangeDetector:
    """
    Selects semantically significant frames from a full frame sequence.

    Algorithm (per consecutive frame pair):
      1. Grayscale MSE  — captures pixel-level brightness/structure change.
      2. Dense Farneback optical-flow magnitude mean — captures motion.
      3. Combined score = mse_weight * norm_mse + flow_weight * norm_flow.
      4. A frame is a keyframe when combined_score > combined_threshold
         OR it is forced (first / last frame, hard scene cut).
      5. A minimum interval between keyframes is enforced to avoid bursts.
    """

    # MSE above this is treated as a hard scene cut regardless of flow
    _HARD_CUT_MSE = 5000.0
    # Normalisation caps (values above are clamped to 1.0)
    _MSE_NORM_CAP = 3000.0
    _FLOW_NORM_CAP = 15.0

    def __init__(self, config: dict):
        cd = config.get("change_detection", {})
        self.mse_threshold: float = cd.get("mse_threshold", 500.0)
        self.flow_threshold: float = cd.get("flow_threshold", 2.0)
        self.combined_threshold: float = cd.get("combined_threshold", 0.3)
        self.mse_weight: float = cd.get("mse_weight", 0.4)
        self.flow_weight: float = cd.get("flow_weight", 0.6)
        self.min_interval: int = cd.get("min_keyframe_interval", 5)
        self.always_endpoints: bool = cd.get("always_include_first_last", True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def detect_keyframes(self, frames: List[Frame]) -> List[KeyFrame]:
        """Return the subset of frames that represent significant change."""
        if not frames:
            return []
        if len(frames) == 1:
            return [KeyFrame(frame=frames[0], change_score=0.0)]

        keyframes: List[KeyFrame] = []
        last_kf_index = -self.min_interval  # allow first frame to be selected

        for i, frame in enumerate(frames):
            if i == 0:
                if self.always_endpoints:
                    keyframes.append(KeyFrame(frame=frame, change_score=0.0))
                    last_kf_index = 0
                continue

            prev = frames[i - 1]
            mse = self._compute_mse(prev, frame)
            flow_mag = self._compute_flow_magnitude(prev, frame)
            score = self._combined_score(mse, flow_mag)
            is_cut = mse >= self._HARD_CUT_MSE

            frames_since_last = i - last_kf_index
            if frames_since_last < self.min_interval and not is_cut:
                continue

            if score >= self.combined_threshold or is_cut:
                keyframes.append(KeyFrame(frame=frame, change_score=score, is_scene_cut=is_cut))
                last_kf_index = i

        # Always include last frame
        if self.always_endpoints and (not keyframes or keyframes[-1].frame.index != frames[-1].index):
            keyframes.append(KeyFrame(frame=frames[-1], change_score=0.0))

        return keyframes

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_gray(frame: Frame) -> np.ndarray:
        return cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _compute_mse(self, f1: Frame, f2: Frame) -> float:
        g1 = self._to_gray(f1)
        g2 = self._to_gray(f2)
        return float(np.mean((g1 - g2) ** 2))

    def _compute_flow_magnitude(self, f1: Frame, f2: Frame) -> float:
        g1 = self._to_gray(f1).astype(np.uint8)
        g2 = self._to_gray(f2).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(magnitude))

    def _combined_score(self, mse: float, flow: float) -> float:
        norm_mse = min(mse / self._MSE_NORM_CAP, 1.0)
        norm_flow = min(flow / self._FLOW_NORM_CAP, 1.0)
        return self.mse_weight * norm_mse + self.flow_weight * norm_flow
