"""
FastAPI service (no frontend) exposing:

  POST /api/swap
      Body: application/xml, e.g.

        <SwapRequest>
          <SiteId>site123</SiteId>
          <OriginalSource>video1.mp4</OriginalSource>
          <SwapSource>image2.jpg</SwapSource>
          <TargetSource>dipika.jpg</TargetSource>   <!-- optional -->
        </SwapRequest>

      SiteId identifies which site/tenant this job belongs to. It's stored
      alongside the job's status record in Redis and echoed back in every
      API response for this job (POST /api/swap and GET /api/swap/{job_id}).

      OriginalSource and SwapSource are filenames expected to already exist
      in settings.uploads_dir (the shared folder the website/mobile server
      drops files into). Whether this is an image-on-image or
      image-on-video swap is decided purely by the extension of
      OriginalSource.

      TargetSource is optional and matters when OriginalSource has more
      than one face in it (e.g. a video with 2 men and 1 woman). It's a
      reference photo of the specific person whose face should be
      replaced:
        - TargetSource given: whichever face in OriginalSource matches that
          person (by face recognition) gets swapped; everyone else is left
          alone.
        - TargetSource omitted: the first female face (reading left to
          right) is swapped by default.
      See app/core/face_engine.py:select_target_face() for the matching
      logic, and FACE_MATCH_THRESHOLD in .env for tuning it.

      This endpoint only validates the request and hands the actual swap
      off to a queue — it does not perform the swap itself. Once the
      payload has been validated (files exist, are the right type, and are
      within the configured size limits), a job_id is generated, a status
      record is written to Redis under it with status "Starting", and a
      job message is published to RabbitMQ (see app/broker.py). The
      response comes back immediately (202 Accepted) with job_id — it does
      not wait for the swap to finish. job_id is the only identifier for
      the job; there is no caller-supplied id, so the caller must hang on
      to the job_id from the response to check status later, and each POST
      always creates a brand new job (no de-dup).

      app/worker.py is the separate, long-running process that actually
      consumes jobs off the queue, runs the swap, and updates the job's
      status in Redis ("In progress" -> "Completed"/"Failed"). For video
      jobs, it also updates the message field with rough percent-complete
      as it goes (see app/worker.py's progress callback).

  GET /api/swap/{job_id}
      Returns the job's current status as a single JSON response and
      closes immediately — a plain snapshot, not a stream. Poll this as
      often as you like to watch a job's progress.

Run the API with:     uvicorn app.main:app --reload --port 8000
Run the worker with:  python -m app.worker
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

from fastapi import Body, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.broker import BrokerError, SwapJobMessage, publish_swap_job
from app.config import settings
from app.schemas import SwapAccepted, SwapStatus
from app.store import STATUS_STARTING, RequestStoreError, get_store
from app.utils import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    is_image_filename,
    is_video_filename,
    safe_filename,
    validate_image_path,
    validate_video_path,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faceswap.main")

app = FastAPI(title="Face Swap Service API")


def _parse_swap_request(xml_body: str) -> Tuple[str, str, str, Optional[str]]:
    """Parse and sanity-check the XML payload. Returns (site_id, original_source, swap_source, target_source)."""
    try:
        root = ET.fromstring(xml_body)
    except ET.ParseError as exc:
        raise HTTPException(status_code=400, detail="Invalid XML payload.") from exc

    if root.tag != "SwapRequest":
        raise HTTPException(
            status_code=400,
            detail="Invalid XML payload. Root element must be <SwapRequest>.",
        )

    site_id = (root.findtext("SiteId") or "").strip()
    original_source = safe_filename(root.findtext("OriginalSource"))
    swap_source = safe_filename(root.findtext("SwapSource"))
    target_source = safe_filename(root.findtext("TargetSource"))  # optional

    if not site_id:
        raise HTTPException(status_code=400, detail="Missing SiteId.")
    if not original_source:
        raise HTTPException(status_code=400, detail="Missing OriginalSource.")
    if not swap_source:
        raise HTTPException(status_code=400, detail="Missing SwapSource.")

    return site_id, original_source, swap_source, target_source


def _resolve_upload_path(filename: str, label: str) -> Path:
    path = settings.uploads_dir / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"{label} not found in uploads folder: {filename}")
    return path


def _media_type_for(original_name: str) -> str:
    if is_video_filename(original_name):
        return "video"
    if is_image_filename(original_name):
        return "image"
    supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
    raise HTTPException(
        status_code=400,
        detail=f"'{original_name}' is not a supported image or video type ({supported}).",
    )


def _validate_sources(
    original_path: Path,
    face_path: Path,
    media_type: str,
    target_path: Optional[Path],
) -> None:
    if media_type == "video":
        validate_video_path(original_path, settings.max_video_mb, label="OriginalSource")
    else:
        validate_image_path(original_path, settings.max_image_mb, label="OriginalSource")
    validate_image_path(face_path, settings.max_image_mb, label="SwapSource")
    if target_path is not None:
        validate_image_path(target_path, settings.max_image_mb, label="TargetSource")


# --------------------------------------------------------------------------- #
# POST /api/swap — validate the payload, enqueue the job, return immediately.
# The actual swap happens in app/worker.py.
# --------------------------------------------------------------------------- #
@app.post("/api/swap", response_model=SwapAccepted, status_code=202)
async def api_swap(
    xml_body: str = Body(
        ...,
        media_type="application/xml",
        example="""<SwapRequest>
  <SiteId>site123</SiteId>
  <OriginalSource>video1.mp4</OriginalSource>
  <SwapSource>image2.jpg</SwapSource>
  <TargetSource>dipika.jpg</TargetSource>
</SwapRequest>""",
    ),
):
    site_id, original_name, swap_name, target_name = _parse_swap_request(xml_body)
    media_type = _media_type_for(original_name)

    original_path = _resolve_upload_path(original_name, "OriginalSource")
    face_path = _resolve_upload_path(swap_name, "SwapSource")
    target_path = _resolve_upload_path(target_name, "TargetSource") if target_name else None
    _validate_sources(original_path, face_path, media_type, target_path)

    job_id = str(uuid.uuid4())
    store = get_store()

    try:
        await run_in_threadpool(
            store.create, job_id, site_id, original_name, swap_name, media_type, target_name
        )
    except RequestStoreError as exc:
        logger.exception("Status cache unavailable")
        raise HTTPException(status_code=503, detail="Status cache unavailable. Try again shortly.") from exc

    job_message = SwapJobMessage(
        job_id=job_id,
        site_id=site_id,
        original_source=original_name,
        swap_source=swap_name,
        media_type=media_type,
        target_source=target_name,
    )
    try:
        await run_in_threadpool(publish_swap_job, job_message)
    except BrokerError as exc:
        logger.exception("Failed to publish swap job to RabbitMQ")
        # The status record already says "Starting" but nothing was actually
        # queued — correct it so a status check doesn't look like a real job
        # is in flight when nothing was.
        await run_in_threadpool(
            store.update_status, job_id, "Failed", f"Could not queue job: {exc}"
        )
        raise HTTPException(status_code=503, detail="Could not queue swap job. Try again shortly.") from exc

    return SwapAccepted(job_id=job_id, site_id=site_id, media_type=media_type, status=STATUS_STARTING)


# --------------------------------------------------------------------------- #
# GET /api/swap/{job_id} — one-shot status snapshot. Returns immediately
# with whatever is currently in Redis and closes; call it again to see the
# next update (e.g. video jobs' message field advances roughly every 5%).
# --------------------------------------------------------------------------- #
@app.get("/api/swap/{job_id}", response_model=SwapStatus)
async def get_swap_status(job_id: str):
    store = get_store()
    try:
        record = await run_in_threadpool(store.get, job_id)
    except RequestStoreError as exc:
        logger.exception("Status cache unavailable")
        raise HTTPException(status_code=503, detail="Status cache unavailable. Try again shortly.") from exc

    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    return SwapStatus(**asdict(record))
