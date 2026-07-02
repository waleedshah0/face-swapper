"""
FastAPI service exposing a single XML-based validation endpoint for future swap processing.

Run with: uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from fastapi import Body, FastAPI, HTTPException

from app.config import settings
from app.schemas import SwapValidationResponse
from app.utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, is_image_filename, is_video_filename, safe_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faceswap.main")

app = FastAPI(title="Face Swap Service API")


def _read_required_tag(root: ET.Element, tag_name: str) -> str:
    element = root.find(tag_name)
    if element is None or element.text is None:
        raise HTTPException(status_code=400, detail=f"Missing {tag_name}.")

    value = element.text.strip()
    if not value:
        raise HTTPException(status_code=400, detail=f"{tag_name} cannot be empty.")

    return value


def _parse_swap_request(xml_body: str) -> tuple[str, str, Optional[str]]:
    try:
        root = ET.fromstring(xml_body)
    except ET.ParseError as exc:
        raise HTTPException(status_code=400, detail="Invalid XML payload.") from exc

    if root.tag != "SwapRequest":
        raise HTTPException(status_code=400, detail="Invalid XML payload: expected root element <SwapRequest>.")

    trans_id = safe_filename(_read_required_tag(root, "TransId"))
    original_source = safe_filename(_read_required_tag(root, "OriginalSource"))
    swap_source = safe_filename(_read_required_tag(root, "SwapSource"))

    if not original_source:
        raise HTTPException(status_code=400, detail="OriginalSource cannot be empty.")
    if not swap_source:
        raise HTTPException(status_code=400, detail="SwapSource cannot be empty.")

    if not is_image_filename(swap_source):
        raise HTTPException(status_code=400, detail="SwapSource must be an image file.")

    if not is_video_filename(original_source) and not is_image_filename(original_source):
        supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"OriginalSource must be a supported image or video file ({supported}).",
        )

    return original_source, swap_source, trans_id


@app.post("/api/swap", response_model=SwapValidationResponse)
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

    if not original_path.exists():
        raise HTTPException(status_code=404, detail=f"OriginalSource file not found in uploads folder: {original_name}")
    if not face_path.exists():
        raise HTTPException(status_code=404, detail=f"SwapSource file not found in uploads folder: {swap_name}")

    return SwapValidationResponse(
        status="success",
        message="Files found successfully. Face swap processing is currently in progress and will be available in the next release.",
        original_source=original_name,
        swap_source=swap_name,
        uploads_dir=str(settings.uploads_dir),
    )
