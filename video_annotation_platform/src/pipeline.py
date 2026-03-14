"""Main orchestrator: runs the full annotation pipeline end-to-end."""

from __future__ import annotations

import os
from typing import Optional

import yaml
from tqdm import tqdm

from .action_detector import ActionDetector
from .change_detector import ChangeDetector
from .frame_annotator import EpisodeContext, FrameAnnotator
from .output_formatter import VLAFormatter
from .video_processor import VideoProcessor


class AnnotationPipeline:
    """
    Orchestrates:
      1. Video frame extraction        (VideoProcessor)
      2. Key frame detection           (ChangeDetector)
      3. Frame annotation              (FrameAnnotator)
      4. Action / motion labelling     (ActionDetector)
      5. VLA JSON + image output       (VLAFormatter)
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        video_path: str,
        output_dir: str = "outputs",
        annotator_override: Optional[str] = None,
        sensitivity_override: Optional[float] = None,
        max_frames: Optional[int] = None,
        visualize: bool = True,
    ) -> str:
        """
        Run the full pipeline.

        Parameters
        ----------
        video_path        : path to input MP4 (or other supported video)
        output_dir        : root directory for all outputs
        annotator_override: "claude" | "local" | "auto"  (overrides config)
        sensitivity_override: combined_threshold override (0.0 – 1.0)
        max_frames        : cap on the number of key frames to annotate
        visualize         : whether to save annotated overlay images

        Returns
        -------
        Path to the generated episode JSON file.
        """

        # Apply CLI overrides
        cfg = self._merge_overrides(annotator_override, sensitivity_override, visualize)

        print(f"\n{'='*60}")
        print(f"  Video Annotation Platform — VLA Edition")
        print(f"{'='*60}")
        print(f"  Input  : {video_path}")
        print(f"  Output : {output_dir}")
        print(f"  Annotator : {cfg['annotation']['primary']}")
        print(f"{'='*60}\n")

        # -------------------------------------------------------------- #
        # Step 1: Video ingestion                                          #
        # -------------------------------------------------------------- #
        print("[1/5] Loading video …")
        processor = VideoProcessor(video_path, cfg)
        meta = processor.get_metadata()
        print(f"      {meta.total_frames} frames  |  {meta.fps:.1f} fps  |  "
              f"{meta.duration:.1f}s  |  {meta.width}×{meta.height}")

        # -------------------------------------------------------------- #
        # Step 2: Extract all frames (in memory for small videos;         #
        #         streamed otherwise)                                     #
        # -------------------------------------------------------------- #
        print("[2/5] Extracting frames …")
        frames = list(tqdm(processor.extract_frames(), total=meta.total_frames,
                           desc="  frames", unit="fr"))
        processor.close()

        # -------------------------------------------------------------- #
        # Step 3: Key frame detection                                      #
        # -------------------------------------------------------------- #
        print("[3/5] Detecting key frames …")
        detector = ChangeDetector(cfg)
        keyframes = detector.detect_keyframes(frames)

        if max_frames and len(keyframes) > max_frames:
            # Keep evenly-spaced subset + always endpoints
            step = len(keyframes) / max_frames
            indices = {int(i * step) for i in range(max_frames)}
            indices.add(0)
            indices.add(len(keyframes) - 1)
            keyframes = [keyframes[i] for i in sorted(indices)]

        print(f"      {len(keyframes)} key frames selected "
              f"({len(keyframes)/meta.total_frames*100:.1f}% of total)")

        # -------------------------------------------------------------- #
        # Step 4: Frame annotation + action detection                     #
        # -------------------------------------------------------------- #
        print("[4/5] Annotating key frames …")
        annotator = FrameAnnotator(cfg)
        action_det = ActionDetector(cfg)
        context = EpisodeContext()
        annotations = []
        actions = []

        for i, kf in enumerate(tqdm(keyframes, desc="  annotating", unit="frame")):
            ann = annotator.annotate_frame(kf, context)
            context.add(ann.language_annotation)

            prev_kf = keyframes[i - 1] if i > 0 else None
            act = action_det.detect_action(prev_kf, kf)

            annotations.append(ann)
            actions.append(act)

        # Episode-level task summary
        print("      Inferring episode task …")
        task = annotator.infer_episode_task(annotations)
        annotation_model = (
            cfg["annotation"]["claude_model"]
            if cfg["annotation"]["primary"] in ("claude", "auto")
            else "local (BLIP-2 + YOLOv8)"
        )

        # -------------------------------------------------------------- #
        # Step 5: Save outputs                                             #
        # -------------------------------------------------------------- #
        print("[5/5] Saving outputs …")
        formatter = VLAFormatter(output_dir, cfg)
        episode = formatter.format_episode(
            video_meta=meta,
            keyframes=keyframes,
            annotations=annotations,
            actions=actions,
            language_instruction=task,
            annotation_model=annotation_model,
        )
        json_path = formatter.save(episode, keyframes, annotations)

        print(f"\n  Done!  Episode saved to: {json_path}")
        print(f"  Steps : {len(episode.steps)}")
        print(f"  Task  : {task}\n")
        return json_path

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_config(path: Optional[str]) -> dict:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        # Sensible defaults when no config file is provided
        return {
            "video": {"resize_frames": True, "target_size": [640, 480],
                       "supported_formats": ["mp4", "avi", "mov", "mkv"]},
            "change_detection": {"combined_threshold": 0.3, "mse_weight": 0.4,
                                  "flow_weight": 0.6, "min_keyframe_interval": 5,
                                  "always_include_first_last": True},
            "annotation": {"primary": "auto", "claude_model": "claude-sonnet-4-6",
                            "claude_max_tokens": 1024, "claude_image_quality": "high",
                            "local_caption_model": "Salesforce/blip2-opt-2.7b",
                            "local_detection_model": "yolov8n.pt", "local_device": "cpu",
                            "max_objects_per_frame": 10, "api_retry_attempts": 3,
                            "api_retry_delay": 2.0},
            "action": {"use_optical_flow": True, "motion_threshold": 1.5},
            "output": {"save_raw_frames": True, "save_annotated_frames": True,
                        "image_format": "jpg", "image_quality": 95,
                        "json_indent": 2, "episode_prefix": "episode"},
        }

    def _merge_overrides(
        self,
        annotator: Optional[str],
        sensitivity: Optional[float],
        visualize: bool,
    ) -> dict:
        import copy
        cfg = copy.deepcopy(self.config)
        if annotator:
            cfg.setdefault("annotation", {})["primary"] = annotator
        if sensitivity is not None:
            cfg.setdefault("change_detection", {})["combined_threshold"] = sensitivity
        cfg.setdefault("output", {})["save_annotated_frames"] = visualize
        return cfg
