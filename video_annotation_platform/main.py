#!/usr/bin/env python3
"""
Video Annotation Platform — CLI entry point.

Usage examples
--------------
# Annotate with Claude (requires ANTHROPIC_API_KEY):
    python main.py --video robot_demo.mp4 --annotator claude

# Annotate fully offline:
    python main.py --video robot_demo.mp4 --annotator local

# Custom sensitivity + frame cap + visualisation:
    python main.py --video robot_demo.mp4 \\
                   --sensitivity 0.25 \\
                   --max-frames 50 \\
                   --visualize \\
                   --output ./my_outputs
"""

import argparse
import os
import sys

# Allow running directly as `python main.py` from the package directory
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import AnnotationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vap",
        description=(
            "Video Annotation Platform — produce VLA-ready JSON + images "
            "from an MP4 robot task video."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--video", "-v",
        required=True,
        metavar="PATH",
        help="Path to the input video file (MP4, AVI, MOV, MKV).",
    )

    # Optional
    parser.add_argument(
        "--output", "-o",
        default="outputs",
        metavar="DIR",
        help="Output directory (default: ./outputs).",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        metavar="PATH",
        help="Path to config.yaml (default: configs/config.yaml relative to this file).",
    )
    parser.add_argument(
        "--annotator", "-a",
        choices=["claude", "local", "auto"],
        default=None,
        help=(
            "Annotation engine. 'claude' uses Claude Vision API (needs ANTHROPIC_API_KEY). "
            "'local' uses BLIP-2 + YOLOv8 offline. 'auto' picks Claude if key is set (default)."
        ),
    )
    parser.add_argument(
        "--sensitivity", "-s",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Frame-change sensitivity threshold 0.0–1.0. Lower = more key frames. "
            "Default from config (0.3)."
        ),
    )
    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of key frames to annotate (useful for long videos).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Save annotated overlay images alongside raw frames.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve config path: prefer --config arg, then package-default
    config_path = args.config
    if config_path is None:
        default_cfg = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
        if os.path.isfile(default_cfg):
            config_path = default_cfg

    pipeline = AnnotationPipeline(config_path=config_path)
    json_path = pipeline.run(
        video_path=args.video,
        output_dir=args.output,
        annotator_override=args.annotator,
        sensitivity_override=args.sensitivity,
        max_frames=args.max_frames,
        visualize=args.visualize,
    )

    print(f"Episode JSON : {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
