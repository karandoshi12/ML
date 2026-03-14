"""VLA-compatible JSON + image output formatter."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .action_detector import ActionLabel
from .change_detector import KeyFrame
from .frame_annotator import DetectedObject, FrameAnnotation
from .video_processor import VideoMetadata

# ---------------------------------------------------------------------------
# VLA Episode data model
# ---------------------------------------------------------------------------

@dataclass
class VLAStep:
    step_id: int
    timestamp: float
    frame_index: int
    image_path: str
    annotated_image_path: Optional[str]
    is_keyframe: bool
    change_score: float
    observation: Dict
    action: Dict
    language_annotation: str


@dataclass
class VLAEpisode:
    schema_version: str
    metadata: Dict
    steps: List[VLAStep]


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class VLAFormatter:
    """Saves annotated frames and writes a VLA-compatible JSON episode file."""

    SCHEMA_VERSION = "1.0"
    PLATFORM = "video-annotation-platform/1.0"

    def __init__(self, output_dir: str, config: dict):
        self.output_dir = os.path.abspath(output_dir)
        self.config = config
        out_cfg = config.get("output", {})
        self.save_raw: bool = out_cfg.get("save_raw_frames", True)
        self.save_annotated: bool = out_cfg.get("save_annotated_frames", True)
        self.img_fmt: str = out_cfg.get("image_format", "jpg")
        self.img_quality: int = out_cfg.get("image_quality", 95)
        self.json_indent: int = out_cfg.get("json_indent", 2)
        self.episode_prefix: str = out_cfg.get("episode_prefix", "episode")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def format_episode(
        self,
        video_meta: VideoMetadata,
        keyframes: List[KeyFrame],
        annotations: List[FrameAnnotation],
        actions: List[ActionLabel],
        language_instruction: str,
        annotation_model: str,
    ) -> VLAEpisode:
        """Build the in-memory VLAEpisode structure (does not write to disk)."""
        episode_id = f"{self.episode_prefix}_{uuid.uuid4().hex[:8]}"
        frames_dir = os.path.join(self.output_dir, episode_id, "frames")

        steps: List[VLAStep] = []
        for i, (kf, ann, act) in enumerate(zip(keyframes, annotations, actions)):
            rel_raw, rel_ann = self._image_paths(episode_id, i)
            step = VLAStep(
                step_id=i,
                timestamp=round(kf.frame.timestamp, 4),
                frame_index=kf.frame.index,
                image_path=rel_raw,
                annotated_image_path=rel_ann if self.save_annotated else None,
                is_keyframe=True,
                change_score=round(kf.change_score, 4),
                observation={
                    "scene_description": ann.scene_description,
                    "objects": [
                        {
                            "label": obj.label,
                            "bbox": obj.bbox,
                            "confidence": obj.confidence,
                        }
                        for obj in ann.objects
                    ],
                    "spatial_relations": ann.spatial_relations,
                    "motion_description": ann.motion_description,
                },
                action={
                    "action_type": act.action_type,
                    "description": act.description,
                    "motion_vector": act.motion_vector,
                    "confidence": act.confidence,
                },
                language_annotation=ann.language_annotation,
            )
            steps.append(step)

        metadata = {
            "episode_id": episode_id,
            "source_video": os.path.basename(video_meta.path),
            "fps": video_meta.fps,
            "duration_seconds": round(video_meta.duration, 3),
            "total_frames": video_meta.total_frames,
            "key_frames_count": len(keyframes),
            "language_instruction": language_instruction,
            "annotation_model": annotation_model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "platform": self.PLATFORM,
            "resolution": {"width": video_meta.width, "height": video_meta.height},
        }

        return VLAEpisode(
            schema_version=self.SCHEMA_VERSION,
            metadata=metadata,
            steps=steps,
        )

    def save(
        self,
        episode: VLAEpisode,
        keyframes: List[KeyFrame],
        annotations: List[FrameAnnotation],
    ) -> str:
        """Write frames + JSON to disk. Returns path to the episode JSON."""
        episode_id = episode.metadata["episode_id"]
        episode_dir = os.path.join(self.output_dir, episode_id)
        frames_dir = os.path.join(episode_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for i, (kf, ann, step) in enumerate(zip(keyframes, annotations, episode.steps)):
            # Save raw frame
            if self.save_raw:
                raw_path = os.path.join(self.output_dir, step.image_path)
                os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                self._write_image(kf.frame.image, raw_path)

            # Save annotated frame
            if self.save_annotated and step.annotated_image_path:
                ann_path = os.path.join(self.output_dir, step.annotated_image_path)
                os.makedirs(os.path.dirname(ann_path), exist_ok=True)
                annotated = self._draw_annotations(kf.frame.image, ann, step)
                self._write_image(annotated, ann_path)

        # Write JSON manifest
        json_path = os.path.join(episode_dir, "episode.json")
        episode_dict = {
            "schema_version": episode.schema_version,
            "metadata": episode.metadata,
            "steps": [self._step_to_dict(s) for s in episode.steps],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(episode_dict, f, indent=self.json_indent, ensure_ascii=False)

        return json_path

    # ------------------------------------------------------------------ #
    # Image helpers                                                        #
    # ------------------------------------------------------------------ #

    def _write_image(self, image_bgr: np.ndarray, path: str):
        if self.img_fmt.lower() in ("jpg", "jpeg"):
            cv2.imwrite(path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.img_quality])
        else:
            cv2.imwrite(path, image_bgr)

    def _draw_annotations(
        self,
        image_bgr: np.ndarray,
        ann: FrameAnnotation,
        step: VLAStep,
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and a caption bar on the frame."""
        out = image_bgr.copy()
        h, w = out.shape[:2]

        # Draw bounding boxes
        for obj in ann.objects:
            bx, by, bw, bh = obj.bbox
            x1 = int(bx / 100 * w)
            y1 = int(by / 100 * h)
            x2 = int((bx + bw) / 100 * w)
            y2 = int((by + bh) / 100 * h)
            color = self._label_color(obj.label)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label_str = f"{obj.label} {obj.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, y1 - lh - 4), (x1 + lw + 2, y1), color, -1)
            cv2.putText(out, label_str, (x1 + 1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Caption bar at bottom
        caption = step.language_annotation[:100]
        bar_h = 28
        cv2.rectangle(out, (0, h - bar_h), (w, h), (20, 20, 20), -1)
        cv2.putText(out, caption, (6, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # Step info overlay (top-left)
        info = f"step={step.step_id}  t={step.timestamp:.2f}s  score={step.change_score:.2f}"
        cv2.putText(out, info, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

        return out

    @staticmethod
    def _label_color(label: str) -> tuple:
        """Deterministic per-label BGR colour."""
        h = hash(label) % (256 ** 3)
        b = h & 0xFF
        g = (h >> 8) & 0xFF
        r = (h >> 16) & 0xFF
        # Ensure minimum brightness
        if b + g + r < 200:
            b, g, r = b + 80, g + 80, r + 80
        return (b % 256, g % 256, r % 256)

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                                #
    # ------------------------------------------------------------------ #

    def _image_paths(self, episode_id: str, step_id: int) -> tuple:
        fname = f"frame_{step_id:04d}.{self.img_fmt}"
        raw = os.path.join(episode_id, "frames", fname)
        ann_fname = f"frame_{step_id:04d}_annotated.{self.img_fmt}"
        ann = os.path.join(episode_id, "frames", ann_fname)
        return raw, ann

    @staticmethod
    def _step_to_dict(step: VLAStep) -> dict:
        return {
            "step_id": step.step_id,
            "timestamp": step.timestamp,
            "frame_index": step.frame_index,
            "image_path": step.image_path,
            "annotated_image_path": step.annotated_image_path,
            "is_keyframe": step.is_keyframe,
            "change_score": step.change_score,
            "observation": step.observation,
            "action": step.action,
            "language_annotation": step.language_annotation,
        }
