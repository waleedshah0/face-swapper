"""
FastAPI service (no frontend) exposing:

  POST /api/swap
      Body: application/xml, e.g.

        <SwapRequest>
          <TransId>12345</TransId>
          <OriginalSource>video1.mp4</OriginalSource>
          <SwapSource>image2.jpg</SwapSource>
        </SwapRequest>

      OriginalSource and SwapSource are filenames expected to already exist
      in settings.uploads_dir (the shared folder the website/mobile server
      drops files into). Whether this is an image-on-image or
      image-on-video swap is decided purely by the extension of
      OriginalSource.

      Both paths are now fully synchronous: the request blocks until the
      swap is completely finished and written to settings.outputs_dir, then
      returns a small JSON confirmation (not the file itself — the calling
      system is expected to read the actual result straight out of the
      shared outputs folder, per the original shared-folder architecture).

      There is no job/polling step any more. Important: for video, this can
      mean the connection stays open for several minutes (much longer with
      the face enhancer enabled — see README/troubleshooting notes). Make
      sure whatever calls this endpoint (and any reverse proxy/load
      balancer in front of it) is configured with a long enough timeout,
      or this will be cut off mid-swap.

Run with:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from fastapi import Body, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.core.face_engine import NoFaceFoundError
from app.core.image_swap import swap_image
from app.core.video_swap import swap_video
from app.schemas import SwapResult
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


def _parse_swap_request(xml_body: str) -> tuple[str, str, Optional[str]]:
    try:
        root = ET.fromstring(xml_body)
    except ET.ParseError as exc:
        raise HTTPException(status_code=400, detail="Invalid XML payload.") from exc

    trans_id = safe_filename(root.findtext("TransId"))
    original_source = safe_filename(root.findtext("OriginalSource"))
    swap_source = safe_filename(root.findtext("SwapSource"))

    if not original_source:
        raise HTTPException(status_code=400, detail="Missing OriginalSource.")
    if not swap_source:
        raise HTTPException(status_code=400, detail="Missing SwapSource.")

    return original_source, swap_source, trans_id


# --------------------------------------------------------------------------- #
# Single swap endpoint. Decides image-vs-video purely from the extension of
# OriginalSource. The actual swap work runs in a thread pool (via
# run_in_threadpool) so this long-running, CPU-bound work doesn't block the
# server's event loop — but the HTTP response to THIS caller still only
# comes back once the swap is fully done, same as a normal synchronous call.
# --------------------------------------------------------------------------- #
@app.post("/api/swap", response_model=SwapResult)
async def api_swap(
    xml_body: str = Body(
        ...,
        media_type="application/xml",
        example="""<SwapRequest>
  <TransId>12345</TransId>
  <OriginalSource>video1.mp4</OriginalSource>
  <SwapSource>image2.jpg</SwapSource>
</SwapRequest>""",
    ),
):
    original_name, swap_name, trans_id = _parse_swap_request(xml_body)

    original_path = settings.uploads_dir / original_name
    face_path = settings.uploads_dir / swap_name

    if is_video_filename(original_name):
        return await _swap_video_sync(original_path, face_path, trans_id)
    elif is_image_filename(original_name):
        return await _swap_image_sync(original_path, face_path, trans_id)
    else:
        supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"'{original_name}' is not a supported image or video type ({supported}).",
        )


async def _swap_image_sync(original_path: Path, face_path: Path, trans_id: Optional[str]) -> SwapResult:
    validate_image_path(original_path, settings.max_image_mb, label="OriginalSource")
    validate_image_path(face_path, settings.max_image_mb, label="SwapSource")

    output_stem = trans_id or original_path.stem
    output_path = settings.outputs_dir / f"{output_stem}.png"

    try:
        await run_in_threadpool(swap_image, original_path, face_path, output_path)
    except NoFaceFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Image swap failed")
        raise HTTPException(status_code=500, detail="Image swap failed. See server logs.")

    return SwapResult(status="success", trans_id=trans_id, media_type="image", output_file=output_path.name)


async def _swap_video_sync(original_path: Path, face_path: Path, trans_id: Optional[str]) -> SwapResult:
    validate_video_path(original_path, settings.max_video_mb, label="OriginalSource")
    validate_image_path(face_path, settings.max_image_mb, label="SwapSource")

    output_stem = trans_id or original_path.stem
    output_path = settings.outputs_dir / f"{output_stem}.mp4"

    try:
        await run_in_threadpool(swap_video, original_path, face_path, output_path)
    except NoFaceFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Video swap failed")
        raise HTTPException(status_code=500, detail="Video swap failed. See server logs.")

    return SwapResult(status="success", trans_id=trans_id, media_type="video", output_file=output_path.name)
