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
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"

# On Vercel, /tmp is the only writable directory. Use it for both outputs
# and job-state files so polling survives across stateless Lambda invocations.
# On a regular server, keep outputs next to the package as before.
_ON_VERCEL = os.environ.get("VAP_ENV") == "vercel"
OUTPUTS_DIR = Path("/tmp/vap_outputs") if _ON_VERCEL else Path(__file__).parent.parent / "outputs"
_JOBS_DIR   = Path("/tmp/vap_jobs")    if _ON_VERCEL else Path(__file__).parent.parent / "outputs" / ".jobs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
_JOBS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Job state helpers
# Persist job state to individual JSON files in _JOBS_DIR so that:
#   - local server: in-process reads work as before
#   - Vercel: polling requests (different Lambda instances) can read state
#     from /tmp which is shared within the same function execution context
# ---------------------------------------------------------------------------
jobs_lock = threading.Lock()

def _job_path(job_id: str) -> Path:
    return _JOBS_DIR / f"{job_id}.json"

def _write_job(job_id: str, data: Dict[str, Any]) -> None:
    with jobs_lock:
        tmp = _job_path(job_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(_job_path(job_id))

def _update_job(job_id: str, update: Dict[str, Any]) -> None:
    with jobs_lock:
        path = _job_path(job_id)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data.update(update)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)

def _read_job(job_id: str) -> Dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

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
    """Runs in a background thread. Persists state via _update_job()."""
    _update_job(job_id, {"status": "running"})

    # On Vercel force Claude-only (BLIP-2/YOLO are not installed)
    if _ON_VERCEL and annotator == "auto":
        annotator = "claude"

    output_dir = str(OUTPUTS_DIR / job_id)

    try:
        pipeline = _get_pipeline()
        pipeline.config.setdefault("change_detection", {})["combined_threshold"] = sensitivity
        pipeline.config.setdefault("annotation", {})["primary"] = annotator
        pipeline.config.setdefault("output", {})["save_annotated_frames"] = True

        json_path = pipeline.run(
            video_path=video_path,
            output_dir=output_dir,
            annotator_override=annotator,
            sensitivity_override=sensitivity,
            max_frames=max_frames if max_frames > 0 else None,
            visualize=True,
        )

        with open(json_path) as f:
            episode = json.load(f)

        total_steps = len(episode.get("steps", []))
        _update_job(job_id, {
            "status": "done",
            "manifest_path": json_path,
            "episode_dir": output_dir,
            "progress": total_steps,
            "total": total_steps,
        })

    except Exception as exc:
        _update_job(job_id, {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
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

    _write_job(job_id, {
        "status": "pending",
        "progress": 0,
        "total": 0,
        "filename": video.filename,
        "manifest_path": None,
        "error": None,
    })

    # Run pipeline in a background thread so the HTTP response returns
    # immediately and the client can poll /status/{job_id}.
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
    job = _read_job(job_id)
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
    job = _read_job(job_id)
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
    job_count = len(list(_JOBS_DIR.glob("*.json"))) if _JOBS_DIR.exists() else 0
    return {"status": "ok", "jobs_count": job_count, "env": "vercel" if _ON_VERCEL else "local"}
