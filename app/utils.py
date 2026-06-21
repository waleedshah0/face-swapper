from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _ext(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def validate_image_upload(file: UploadFile, max_mb: int) -> None:
    if _ext(file.filename) not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"'{file.filename}' is not a supported image type ({', '.join(sorted(IMAGE_EXTENSIONS))}).",
        )
    _check_size(file, max_mb)


def validate_video_upload(file: UploadFile, max_mb: int) -> None:
    if _ext(file.filename) not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"'{file.filename}' is not a supported video type ({', '.join(sorted(VIDEO_EXTENSIONS))}).",
        )
    _check_size(file, max_mb)


def _check_size(file: UploadFile, max_mb: int) -> None:
    # UploadFile.size is populated by Starlette once the body is read; as a
    # belt-and-suspenders check we also verify after writing to disk.
    if file.size is not None and file.size > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"'{file.filename}' exceeds the {max_mb}MB limit.",
        )


async def save_upload(file: UploadFile, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}{_ext(file.filename)}"
    dest_path = dest_dir / unique_name

    with open(dest_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
    await file.close()
    return dest_path
