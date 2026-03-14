"""
Vercel entry point — imports the FastAPI app so Vercel's Python runtime
can discover and serve it as an ASGI serverless function.

Vercel looks for a variable named `app` (ASGI/WSGI) in this file.
"""

import sys
from pathlib import Path

# Make `video_annotation_platform` importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "video_annotation_platform"))

from web.app import app  # noqa: F401  — re-exported for Vercel
