"""Frame annotation using Claude Vision API (primary) or BLIP-2 + YOLOv8 (fallback)."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from .change_detector import KeyFrame

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DetectedObject:
    label: str
    bbox: List[float]       # [x%, y%, w%, h%] as % of frame dimensions
    confidence: float


@dataclass
class FrameAnnotation:
    scene_description: str
    objects: List[DetectedObject]
    spatial_relations: List[str]
    motion_description: str
    language_annotation: str
    annotation_source: str          # "claude" | "local"


@dataclass
class EpisodeContext:
    """Accumulated context passed to the annotator for each frame."""
    prior_annotations: List[str] = field(default_factory=list)

    def summary(self, max_entries: int = 3) -> str:
        if not self.prior_annotations:
            return "No prior frames annotated yet."
        recent = self.prior_annotations[-max_entries:]
        return " → ".join(recent)

    def add(self, annotation: str):
        self.prior_annotations.append(annotation)


# ---------------------------------------------------------------------------
# Main annotator
# ---------------------------------------------------------------------------

class FrameAnnotator:
    """
    Annotates key frames for VLA training data.

    Strategy:
      - primary="claude"  → Claude Vision API only
      - primary="local"   → BLIP-2 + YOLOv8 only
      - primary="auto"    → Claude if ANTHROPIC_API_KEY is set, else local
    """

    _SYSTEM_PROMPT = (
        "You are an expert robotics data annotator producing training data for "
        "Vision-Language-Action (VLA) models.\n"
        "Analyze frames from robot task videos with precision. For each frame provide:\n"
        "1. scene_description: factual description of the environment and all visible objects.\n"
        "2. objects: list each salient object with approximate bounding box as percentages "
        "[x%, y%, w%, h%] of frame size, and confidence 0-1.\n"
        "3. spatial_relations: key positional/relational statements between objects "
        "(e.g. 'gripper above red block').\n"
        "4. motion_description: what movement or action is occurring or has just completed.\n"
        "5. language_annotation: one concise natural-language instruction a robot should "
        "execute at this moment, suitable for VLA training.\n\n"
        "Focus on: robot end-effectors, task-relevant objects, state changes. "
        "Be factual and concise. Respond ONLY with valid JSON."
    )

    _USER_PROMPT_TPL = (
        "Frame {frame_index} at t={timestamp:.3f}s (change_score={change_score:.2f}).\n"
        "Task context so far: {episode_context}\n\n"
        "Annotate this frame for VLA training. "
        "Return a JSON object with keys: scene_description (str), "
        "objects (list of {{label, bbox:[x%,y%,w%,h%], confidence}}), "
        "spatial_relations (list of str), "
        "motion_description (str), "
        "language_annotation (str)."
    )

    def __init__(self, config: dict):
        self.config = config
        ann = config.get("annotation", {})
        self.primary: str = ann.get("primary", "auto")
        self.claude_model: str = ann.get("claude_model", "claude-sonnet-4-6")
        self.claude_max_tokens: int = ann.get("claude_max_tokens", 1024)
        self.image_quality: str = ann.get("claude_image_quality", "high")
        self.max_objects: int = ann.get("max_objects_per_frame", 10)
        self.retry_attempts: int = ann.get("api_retry_attempts", 3)
        self.retry_delay: float = ann.get("api_retry_delay", 2.0)
        self.local_device: str = ann.get("local_device", "cpu")
        self.local_detection_model: str = ann.get("local_detection_model", "yolov8n.pt")
        self.local_caption_model: str = ann.get("local_caption_model", "Salesforce/blip2-opt-2.7b")

        self._claude_client = None
        self._yolo_model = None
        self._blip_processor = None
        self._blip_model = None

        # Resolve effective strategy
        self._use_claude = self._resolve_strategy()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def annotate_frame(
        self,
        keyframe: KeyFrame,
        context: EpisodeContext,
    ) -> FrameAnnotation:
        """Annotate a single key frame, falling back to local models if needed."""
        if self._use_claude:
            try:
                return self._annotate_with_claude(keyframe, context)
            except Exception as exc:
                print(f"[FrameAnnotator] Claude failed ({exc}), falling back to local models.")
        return self._annotate_with_local(keyframe)

    def infer_episode_task(self, annotations: List[FrameAnnotation]) -> str:
        """Summarise all language_annotations into a single episode-level task string."""
        if not annotations:
            return "Unknown task"

        unique = list(dict.fromkeys(a.language_annotation for a in annotations))
        if self._use_claude and self._claude_client is not None:
            try:
                return self._episode_summary_claude(unique)
            except Exception:
                pass

        # Simple fallback: return the most common annotation
        from collections import Counter
        counts = Counter(a.language_annotation for a in annotations)
        return counts.most_common(1)[0][0]

    # ------------------------------------------------------------------ #
    # Claude annotation                                                    #
    # ------------------------------------------------------------------ #

    def _annotate_with_claude(
        self, keyframe: KeyFrame, context: EpisodeContext
    ) -> FrameAnnotation:
        client = self._get_claude_client()
        b64 = self._encode_image_b64(keyframe.frame.image)
        prompt = self._USER_PROMPT_TPL.format(
            frame_index=keyframe.frame.index,
            timestamp=keyframe.frame.timestamp,
            change_score=keyframe.change_score,
            episode_context=context.summary(),
        )

        last_exc = None
        for attempt in range(self.retry_attempts):
            try:
                response = client.messages.create(
                    model=self.claude_model,
                    max_tokens=self.claude_max_tokens,
                    system=self._SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": b64,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
                raw = response.content[0].text.strip()
                return self._parse_claude_response(raw)
            except Exception as exc:
                last_exc = exc
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        raise last_exc

    def _parse_claude_response(self, raw: str) -> FrameAnnotation:
        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data: Dict[str, Any] = json.loads(raw)

        objects = []
        for obj in data.get("objects", [])[: self.max_objects]:
            objects.append(DetectedObject(
                label=str(obj.get("label", "object")),
                bbox=obj.get("bbox", [0, 0, 10, 10]),
                confidence=float(obj.get("confidence", 0.5)),
            ))

        return FrameAnnotation(
            scene_description=str(data.get("scene_description", "")),
            objects=objects,
            spatial_relations=list(data.get("spatial_relations", [])),
            motion_description=str(data.get("motion_description", "")),
            language_annotation=str(data.get("language_annotation", "")),
            annotation_source="claude",
        )

    def _episode_summary_claude(self, unique_annotations: List[str]) -> str:
        client = self._get_claude_client()
        bullet_list = "\n".join(f"- {a}" for a in unique_annotations[:20])
        response = client.messages.create(
            model=self.claude_model,
            max_tokens=128,
            messages=[{
                "role": "user",
                "content": (
                    "The following are per-frame language annotations from a robot task video:\n"
                    f"{bullet_list}\n\n"
                    "Write ONE concise sentence (max 15 words) summarising the overall task "
                    "the robot is performing. Reply with ONLY the sentence."
                ),
            }],
        )
        return response.content[0].text.strip()

    # ------------------------------------------------------------------ #
    # Local annotation (BLIP-2 + YOLOv8)                                  #
    # ------------------------------------------------------------------ #

    def _annotate_with_local(self, keyframe: KeyFrame) -> FrameAnnotation:
        caption = self._blip_caption(keyframe.frame.image)
        objects = self._yolo_detect(keyframe.frame.image)
        spatial = self._compute_spatial_relations(objects, keyframe.frame.image.shape)
        return FrameAnnotation(
            scene_description=caption,
            objects=objects,
            spatial_relations=spatial,
            motion_description="",  # cannot be computed from single frame alone
            language_annotation=caption,
            annotation_source="local",
        )

    def _blip_caption(self, image_bgr: np.ndarray) -> str:
        proc, model = self._get_blip()
        pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = proc(images=pil_img, return_tensors="pt").to(self.local_device)
        import torch
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    def _yolo_detect(self, image_bgr: np.ndarray) -> List[DetectedObject]:
        model = self._get_yolo()
        h, w = image_bgr.shape[:2]
        results = model(image_bgr, verbose=False)
        objects: List[DetectedObject] = []
        for box in results[0].boxes[: self.max_objects]:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            bx = round(float(x1 / w * 100), 1)
            by = round(float(y1 / h * 100), 1)
            bw = round(float((x2 - x1) / w * 100), 1)
            bh = round(float((y2 - y1) / h * 100), 1)
            label = results[0].names[int(box.cls[0])]
            conf = float(box.conf[0])
            objects.append(DetectedObject(label=label, bbox=[bx, by, bw, bh], confidence=round(conf, 3)))
        return objects

    @staticmethod
    def _compute_spatial_relations(
        objects: List[DetectedObject], shape: tuple
    ) -> List[str]:
        """Generate simple spatial relation strings from bounding boxes."""
        relations: List[str] = []
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                ax_c = a.bbox[0] + a.bbox[2] / 2
                ay_c = a.bbox[1] + a.bbox[3] / 2
                bx_c = b.bbox[0] + b.bbox[2] / 2
                by_c = b.bbox[1] + b.bbox[3] / 2
                dx = ax_c - bx_c
                dy = ay_c - by_c
                if abs(dy) > abs(dx):
                    rel = "above" if dy < 0 else "below"
                else:
                    rel = "left of" if dx < 0 else "right of"
                relations.append(f"{a.label} {rel} {b.label}")
        return relations[:8]  # keep concise

    # ------------------------------------------------------------------ #
    # Lazy-loaded model accessors                                          #
    # ------------------------------------------------------------------ #

    def _get_claude_client(self):
        if self._claude_client is None:
            import anthropic
            self._claude_client = anthropic.Anthropic()
        return self._claude_client

    def _get_yolo(self):
        if self._yolo_model is None:
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.local_detection_model)
        return self._yolo_model

    def _get_blip(self):
        if self._blip_processor is None or self._blip_model is None:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch
            self._blip_processor = Blip2Processor.from_pretrained(self.local_caption_model)
            self._blip_model = Blip2ForConditionalGeneration.from_pretrained(
                self.local_caption_model,
                torch_dtype=torch.float16 if self.local_device == "cuda" else torch.float32,
            ).to(self.local_device)
        return self._blip_processor, self._blip_model

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _resolve_strategy(self) -> bool:
        """Return True if Claude should be used as the primary annotator."""
        if self.primary == "claude":
            return True
        if self.primary == "local":
            return False
        # auto: use Claude if API key is available
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    @staticmethod
    def _encode_image_b64(image_bgr: np.ndarray, quality: int = 85) -> str:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        import io
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
