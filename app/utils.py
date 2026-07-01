from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _ext(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def is_image_filename(filename: str) -> bool:
    return _ext(filename) in IMAGE_EXTENSIONS


def is_video_filename(filename: str) -> bool:
    return _ext(filename) in VIDEO_EXTENSIONS


def safe_filename(value: str | None) -> str | None:
    """Return a sanitized filename without directory traversal components."""
    if not value:
        return None
    return Path(value.strip()).name


def require_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label} not found in uploads folder: {path.name}")


def validate_image_path(path: Path, max_mb: int, label: str = "Image") -> None:
    require_file_exists(path, label)
    if not is_image_filename(path.name):
        raise HTTPException(
            status_code=400,
            detail=f"{label} '{path.name}' is not a supported image type ({', '.join(sorted(IMAGE_EXTENSIONS))}).",
        )
    _check_size_on_disk(path, max_mb, label)


def validate_video_path(path: Path, max_mb: int, label: str = "Video") -> None:
    require_file_exists(path, label)
    if not is_video_filename(path.name):
        raise HTTPException(
            status_code=400,
            detail=f"{label} '{path.name}' is not a supported video type ({', '.join(sorted(VIDEO_EXTENSIONS))}).",
        )
    _check_size_on_disk(path, max_mb, label)


def _check_size_on_disk(path: Path, max_mb: int, label: str) -> None:
    size_bytes = path.stat().st_size
    if size_bytes > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"{label} '{path.name}' exceeds the {max_mb}MB limit.",
        )
