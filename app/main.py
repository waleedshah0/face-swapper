"""
FastAPI app exposing:
  GET  /                          -> the website (Image / Video tabs)
  POST /api/swap/image            -> synchronous image swap, returns the file
  POST /api/swap/video             -> starts an async video swap job
  GET  /api/jobs/{job_id}          -> poll job status/progress
  GET  /api/jobs/{job_id}/download -> download the finished video

Run with:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.core.face_engine import NoFaceFoundError
from app.core.image_swap import swap_image
from app.core.video_swap import swap_video
from app.jobs import create_job, get_job, update_job
from app.schemas import JobStatus, VideoJobCreated
from app.utils import save_upload, validate_image_upload, validate_video_upload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faceswap.main")

app = FastAPI(title="Face Swap Studio API")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _require_consent(consent: bool) -> None:
    if not consent:
        raise HTTPException(
            status_code=400,
            detail="You must confirm you have the right to use both photos before swapping.",
        )


# --------------------------------------------------------------------------- #
# Image swap — synchronous, usually sub-second to a few seconds on a GPU
# --------------------------------------------------------------------------- #
@app.post("/api/swap/image")
async def api_swap_image(
    original_image: UploadFile,
    face_image: UploadFile,
    consent: bool = Form(...),
):
    _require_consent(consent)
    validate_image_upload(original_image, settings.max_image_mb)
    validate_image_upload(face_image, settings.max_image_mb)

    original_path = await save_upload(original_image, settings.uploads_dir)
    face_path = await save_upload(face_image, settings.uploads_dir)
    output_path = settings.outputs_dir / f"{original_path.stem}_swapped.png"

    try:
        swap_image(original_path, face_path, output_path)
    except NoFaceFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Image swap failed")
        raise HTTPException(status_code=500, detail="Image swap failed. See server logs.")

    return FileResponse(output_path, media_type="image/png", filename=output_path.name)


# --------------------------------------------------------------------------- #
# Video swap — asynchronous job, since this can take a while
# --------------------------------------------------------------------------- #
@app.post("/api/swap/video", response_model=VideoJobCreated)
async def api_swap_video(
    background_tasks: BackgroundTasks,
    original_video: UploadFile,
    face_image: UploadFile,
    consent: bool = Form(...),
):
    _require_consent(consent)
    validate_video_upload(original_video, settings.max_video_mb)
    validate_image_upload(face_image, settings.max_image_mb)

    original_path = await save_upload(original_video, settings.uploads_dir)
    face_path = await save_upload(face_image, settings.uploads_dir)

    job = create_job()
    output_path = settings.outputs_dir / f"{job.id}.mp4"

    background_tasks.add_task(_run_video_job, job.id, original_path, face_path, output_path)
    return VideoJobCreated(job_id=job.id, status=job.status)


def _run_video_job(job_id: str, original_path: Path, face_path: Path, output_path: Path) -> None:
    update_job(job_id, status="processing", progress=0.0)
    try:
        def on_progress(pct: float) -> None:
            update_job(job_id, progress=pct)

        swap_video(original_path, face_path, output_path, progress_cb=on_progress)
        update_job(job_id, status="done", progress=100.0, result_path=str(output_path))
    except NoFaceFoundError as exc:
        update_job(job_id, status="failed", error=str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Video swap job %s failed", job_id)
        update_job(job_id, status="failed", error="Video swap failed. See server logs.")


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def api_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    download_url = f"/api/jobs/{job_id}/download" if job.status == "done" else None
    return JobStatus(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        error=job.error,
        download_url=download_url,
    )


@app.get("/api/jobs/{job_id}/download")
async def api_job_download(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "done" or not job.result_path:
        raise HTTPException(status_code=409, detail=f"Job is not finished yet (status: {job.status}).")

    return FileResponse(job.result_path, media_type="video/mp4", filename=Path(job.result_path).name)
