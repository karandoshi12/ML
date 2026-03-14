# Video Annotation Platform for VLA Models

A production-ready pipeline that converts robot task videos (MP4) into
**VLA-compatible annotated datasets** — ready to train models like
[OpenVLA](https://github.com/openvla/openvla), [π₀](https://www.physicalintelligence.company/), and [RT-2](https://robotics-transformer2.github.io/).

---

## How It Works

```
MP4 Video
   │
   ▼
┌──────────────────┐
│  VideoProcessor  │  Extract every frame with timestamp
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ChangeDetector  │  MSE + Farneback optical flow → pick key frames
└────────┬─────────┘     (skips redundant/duplicate frames)
         │
         ▼
┌──────────────────────────────────────┐
│  FrameAnnotator                      │
│  ┌─────────────┐  ┌────────────────┐ │
│  │ Claude Vision│  │ BLIP-2 + YOLO  │ │  Rich NL + objects + spatial
│  │  (primary)   │  │  (fallback)    │ │  relations per key frame
│  └─────────────┘  └────────────────┘ │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  ActionDetector  │  Dense optical flow → action type + motion vector
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   VLAFormatter   │  JSON episode manifest + annotated images
└──────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
cd video_annotation_platform
pip install -r requirements.txt
```

### 2. Run with Claude Vision (best quality)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python main.py \
  --video robot_demo.mp4 \
  --annotator claude \
  --visualize
```

### 3. Run fully offline (no API key needed)

```bash
python main.py \
  --video robot_demo.mp4 \
  --annotator local \
  --visualize
```

### 4. Full options

```
python main.py --help

  --video PATH       Input video file (MP4, AVI, MOV, MKV)
  --output DIR       Output directory (default: ./outputs)
  --config PATH      Path to config.yaml
  --annotator        claude | local | auto  (default: auto)
  --sensitivity N    Change threshold 0.0–1.0 (lower = more frames, default: 0.3)
  --max-frames N     Cap on key frames to annotate
  --visualize        Save annotated overlay images
```

---

## Output Structure

```
outputs/
└── episode_a1b2c3d4/
    ├── episode.json           ← VLA manifest (all steps, metadata, annotations)
    └── frames/
        ├── frame_0000.jpg           ← raw key frame
        ├── frame_0000_annotated.jpg ← overlay with boxes + caption
        ├── frame_0001.jpg
        ├── frame_0001_annotated.jpg
        └── …
```

### episode.json schema

```json
{
  "schema_version": "1.0",
  "metadata": {
    "episode_id": "episode_a1b2c3d4",
    "source_video": "robot_demo.mp4",
    "fps": 30.0,
    "duration_seconds": 15.3,
    "total_frames": 459,
    "key_frames_count": 23,
    "language_instruction": "Pick up the red block and place it in the bin",
    "annotation_model": "claude-sonnet-4-6",
    "created_at": "2026-03-14T09:00:00Z",
    "platform": "video-annotation-platform/1.0",
    "resolution": {"width": 640, "height": 480}
  },
  "steps": [
    {
      "step_id": 0,
      "timestamp": 0.033,
      "frame_index": 1,
      "image_path": "episode_a1b2c3d4/frames/frame_0000.jpg",
      "annotated_image_path": "episode_a1b2c3d4/frames/frame_0000_annotated.jpg",
      "is_keyframe": true,
      "change_score": 0.72,
      "observation": {
        "scene_description": "Robotic arm above table with red block at centre.",
        "objects": [
          {"label": "robotic_arm", "bbox": [10, 5, 30, 60], "confidence": 0.97},
          {"label": "red_block",   "bbox": [45, 55, 10, 8],  "confidence": 0.93}
        ],
        "spatial_relations": ["arm above table", "gripper open"],
        "motion_description": "Arm descending toward block."
      },
      "action": {
        "action_type": "approach",
        "description": "Move down — motion down (steadily, flow=6.2px)",
        "motion_vector": {"direction": "down", "magnitude": 6.2, "dx": 0.1, "dy": -6.2},
        "confidence": 0.85
      },
      "language_annotation": "Move the arm down to grasp the red block on the table."
    }
  ]
}
```

---

## Configuration (`configs/config.yaml`)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `video` | `resize_frames` | `true` | Resize all frames before processing |
| `video` | `target_size` | `[640, 480]` | Resize target (width × height) |
| `change_detection` | `combined_threshold` | `0.3` | Key frame sensitivity (0–1) |
| `change_detection` | `min_keyframe_interval` | `5` | Min frames between keyframes |
| `annotation` | `primary` | `auto` | `claude` / `local` / `auto` |
| `annotation` | `claude_model` | `claude-sonnet-4-6` | Claude model ID |
| `annotation` | `local_device` | `cpu` | `cpu` or `cuda` for local models |
| `output` | `save_annotated_frames` | `true` | Draw overlay on saved frames |
| `output` | `image_quality` | `95` | JPEG quality for saved images |

---

## VLA Compatibility

The output JSON follows the **episode/step** structure expected by VLA training frameworks:

| Field | VLA meaning |
|-------|-------------|
| `language_instruction` | Task goal for the episode |
| `observation.scene_description` | Language observation |
| `observation.objects` | Structured object state |
| `action.action_type` | Discretised action label |
| `action.motion_vector` | Continuous motion signal |
| `language_annotation` | Per-step instruction (used as conditioning) |

To convert to HDF5 / RLDS / LeRobot format, load `episode.json` and map fields
with the respective dataset converter.

---

## Project Structure

```
video_annotation_platform/
├── src/
│   ├── video_processor.py   # VideoProcessor — frame extraction
│   ├── change_detector.py   # ChangeDetector — MSE + optical flow keyframe selection
│   ├── frame_annotator.py   # FrameAnnotator — Claude + BLIP-2/YOLO
│   ├── action_detector.py   # ActionDetector — motion taxonomy classification
│   ├── output_formatter.py  # VLAFormatter   — JSON + image output
│   └── pipeline.py          # AnnotationPipeline — orchestrator
├── configs/
│   └── config.yaml
├── outputs/                 # Generated episodes (git-ignored)
├── requirements.txt
└── main.py                  # CLI
```
