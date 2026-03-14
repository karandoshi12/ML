"""
FastAPI web server for the Video Annotation Platform.

Run from video_annotation_platform/:
    uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict

# Allow importing src.* from the package root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline import AnnotationPipeline

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Video Annotation Platform", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory job store: {job_id: {status, progress, total, error, manifest_path}}
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Pipeline singleton (loaded once)
# ---------------------------------------------------------------------------

def _get_pipeline() -> AnnotationPipeline:
    cfg_path = str(CONFIG_PATH) if CONFIG_PATH.exists() else None
    return AnnotationPipeline(config_path=cfg_path)


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _run_annotation(
    job_id: str,
    video_path: str,
    sensitivity: float,
    annotator: str,
    max_frames: int,
) -> None:
    """Runs in a background thread. Updates jobs[job_id] as it progresses."""
    with jobs_lock:
        jobs[job_id]["status"] = "running"

    output_dir = str(OUTPUTS_DIR / job_id)

    try:
        pipeline = _get_pipeline()

        # Override config at pipeline level via config merge
        pipeline.config.setdefault("change_detection", {})["combined_threshold"] = sensitivity
        pipeline.config.setdefault("annotation", {})["primary"] = annotator
        pipeline.config.setdefault("output", {})["save_annotated_frames"] = True

        # Wrap pipeline.run so we can intercept progress
        # We monkey-patch tqdm via the env var approach isn't reliable,
        # so instead we add a simple per-step counter via subclassing.
        json_path = pipeline.run(
            video_path=video_path,
            output_dir=output_dir,
            annotator_override=annotator,
            sensitivity_override=sensitivity,
            max_frames=max_frames if max_frames > 0 else None,
            visualize=True,
        )

        # Load the episode JSON to count total steps
        with open(json_path) as f:
            episode = json.load(f)

        total_steps = len(episode.get("steps", []))

        with jobs_lock:
            jobs[job_id].update({
                "status": "done",
                "manifest_path": json_path,
                "episode_dir": output_dir,
                "progress": total_steps,
                "total": total_steps,
            })

    except Exception as exc:
        tb = traceback.format_exc()
        with jobs_lock:
            jobs[job_id].update({
                "status": "error",
                "error": str(exc),
                "traceback": tb,
            })
    finally:
        # Clean up uploaded temp video file
        try:
            os.remove(video_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page UI."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/annotate")
async def annotate(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    sensitivity: float = Form(0.3),
    annotator: str = Form("auto"),
    max_frames: int = Form(0),
):
    """
    Accept a video upload and start the annotation pipeline.

    Returns {job_id} immediately; client polls /status/{job_id}.
    """
    # Validate file type
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(video.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}",
        )

    # Save to a temp file
    tmp_dir = tempfile.mkdtemp(prefix="vap_upload_")
    safe_name = f"upload{ext}"
    video_path = os.path.join(tmp_dir, safe_name)

    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

    job_id = str(uuid.uuid4())

    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "total": 0,
            "filename": video.filename,
            "manifest_path": None,
            "error": None,
        }

    # Run pipeline in background thread (FastAPI BackgroundTasks run in the
    # event loop; we use threading to avoid blocking async routes)
    thread = threading.Thread(
        target=_run_annotation,
        args=(job_id, video_path, sensitivity, annotator, max_frames),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.get("/status/{job_id}")
async def status(job_id: str):
    """Poll job status. Returns {status, progress, total, error?}."""
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JSONResponse({
        "status": job["status"],
        "progress": job.get("progress", 0),
        "total": job.get("total", 0),
        "filename": job.get("filename", ""),
        "error": job.get("error"),
    })


@app.get("/result/{job_id}")
async def result(job_id: str):
    """Return the full episode.json for a completed job."""
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")

    manifest_path = job.get("manifest_path")
    if not manifest_path or not Path(manifest_path).exists():
        raise HTTPException(status_code=500, detail="Manifest file not found")

    with open(manifest_path, encoding="utf-8") as f:
        episode = json.load(f)

    return JSONResponse(episode)


@app.get("/frames/{job_id}/{filename}")
async def serve_frame(job_id: str, filename: str):
    """Serve an annotated frame image."""
    # Sanitize filename (no path traversal)
    filename = Path(filename).name
    job_output_dir = OUTPUTS_DIR / job_id

    # Search recursively in the episode subdir
    matches = list(job_output_dir.rglob(filename))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Frame not found: {filename}")

    return FileResponse(str(matches[0]))


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "ok", "jobs_count": len(jobs)}
