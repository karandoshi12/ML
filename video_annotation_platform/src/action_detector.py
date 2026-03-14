"""Inter-frame motion analysis and action label classification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .change_detector import KeyFrame

# ---------------------------------------------------------------------------
# Action taxonomy (VLA-compatible)
# ---------------------------------------------------------------------------
ACTION_TAXONOMY = [
    "pick_up",
    "place_down",
    "push",
    "pull",
    "grasp",
    "release",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "move_forward",
    "move_backward",
    "rotate_left",
    "rotate_right",
    "open_gripper",
    "close_gripper",
    "navigate_to",
    "avoid_obstacle",
    "approach",
    "retreat",
    "idle",
]


@dataclass
class ActionLabel:
    action_type: str          # one of ACTION_TAXONOMY
    description: str          # natural-language description
    motion_vector: Dict       # {direction, magnitude, dx, dy}
    confidence: float         # 0.0 – 1.0


class ActionDetector:
    """
    Classifies the dominant action between two consecutive key frames
    by analysing the dense optical-flow field.

    Heuristics (pure CV, no model inference required):
      - Mean flow vector (dx, dy) → cardinal direction + magnitude.
      - Flow magnitude variance → distinguishes purposeful motion from
        camera shake.
      - Dominant motion region (top vs. bottom half) informs arm vs.
        base movement.
    """

    def __init__(self, config: dict):
        act = config.get("action", {})
        self.use_flow: bool = act.get("use_optical_flow", True)
        self.motion_threshold: float = act.get("motion_threshold", 1.5)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def detect_action(
        self,
        prev_kf: Optional[KeyFrame],
        curr_kf: KeyFrame,
    ) -> ActionLabel:
        """Return an ActionLabel describing the transition from prev to curr."""
        if prev_kf is None:
            return ActionLabel(
                action_type="idle",
                description="Initial frame — no prior motion.",
                motion_vector={"direction": "none", "magnitude": 0.0, "dx": 0.0, "dy": 0.0},
                confidence=1.0,
            )

        dx, dy, magnitude, variance = self._analyse_flow(prev_kf, curr_kf)

        if magnitude < self.motion_threshold:
            return ActionLabel(
                action_type="idle",
                description="Minimal motion between frames.",
                motion_vector={"direction": "none", "magnitude": round(magnitude, 3), "dx": round(dx, 3), "dy": round(dy, 3)},
                confidence=0.9,
            )

        direction = self._vector_to_direction(dx, dy)
        action_type = self._classify(dx, dy, magnitude, variance, curr_kf)
        description = self._build_description(action_type, direction, magnitude)
        confidence = min(0.5 + variance / 20.0, 0.95)

        return ActionLabel(
            action_type=action_type,
            description=description,
            motion_vector={
                "direction": direction,
                "magnitude": round(magnitude, 3),
                "dx": round(dx, 3),
                "dy": round(dy, 3),
            },
            confidence=round(confidence, 3),
        )

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _analyse_flow(
        self, prev_kf: KeyFrame, curr_kf: KeyFrame
    ) -> Tuple[float, float, float, float]:
        """Return (mean_dx, mean_dy, magnitude, variance)."""
        g1 = self._gray(prev_kf)
        g2 = self._gray(curr_kf)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        dx = float(np.mean(flow[..., 0]))
        dy = float(np.mean(flow[..., 1]))
        mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
        var = float(np.var(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
        return dx, dy, mag, var

    @staticmethod
    def _gray(kf: KeyFrame) -> np.ndarray:
        return cv2.cvtColor(kf.frame.image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _vector_to_direction(dx: float, dy: float) -> str:
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return "stationary"
        angle = math.degrees(math.atan2(-dy, dx))  # screen coords: y grows down
        # Map angle to 8-directional compass
        dirs = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
        idx = int((angle + 202.5) % 360 / 45)
        return dirs[idx % 8]

    def _classify(
        self, dx: float, dy: float, magnitude: float, variance: float, curr_kf: KeyFrame
    ) -> str:
        h, w = curr_kf.frame.image.shape[:2]

        # Scene cut → approach or retreat based on overall magnitude change
        if curr_kf.is_scene_cut:
            return "navigate_to"

        # Predominantly vertical motion
        if abs(dy) > abs(dx) * 1.5:
            if dy < 0:  # upward in screen = arm lifting
                return "pick_up" if magnitude > 4.0 else "move_up"
            else:       # downward = placing or approaching
                return "place_down" if magnitude > 4.0 else "move_down"

        # Predominantly horizontal motion
        if abs(dx) > abs(dy) * 1.5:
            return "move_left" if dx < 0 else "move_right"

        # Diagonal: high magnitude → approach/retreat
        if magnitude > 6.0:
            return "approach" if dy > 0 else "retreat"

        # Low-variance high-magnitude → purposeful push/pull
        if variance < 5.0 and magnitude > 3.0:
            return "push" if dy > 0 else "pull"

        return "move_forward"

    @staticmethod
    def _build_description(action_type: str, direction: str, magnitude: float) -> str:
        mag_str = "slowly" if magnitude < 3.0 else ("quickly" if magnitude > 8.0 else "steadily")
        readable = action_type.replace("_", " ")
        return f"{readable.capitalize()} — motion {direction} ({mag_str}, flow={magnitude:.1f}px)"
