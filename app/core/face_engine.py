"""
Loads the face detection/analysis model and the face-swap model exactly once
per process, and exposes simple helper functions on top of them.

Models used:
  - buffalo_l (InsightFace FaceAnalysis): face detection, landmarks, embeddings
  - inswapper_128.onnx (InsightFace model zoo): one-shot face identity swap

Why this pair: inswapper_128 is a one-shot swapper, meaning it needs no
per-user training step (unlike DeepFaceLab-style trainers). That makes it
the right fit for an on-demand web service where any two strangers' photos
can be swapped in seconds rather than after hours of training.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger("faceswap.engine")

_lock = threading.Lock()
_face_analyser = None
_face_swapper = None
_face_enhancer = None


class NoFaceFoundError(Exception):
    """Raised when no face can be detected in a supplied image."""


def _load_face_analyser():
    import insightface

    analyser = insightface.app.FaceAnalysis(
        name=settings.face_analyser_name,
        providers=settings.onnx_providers,
    )
    detector_size = settings.face_detector_size
    analyser.prepare(ctx_id=0, det_size=(detector_size, detector_size))
    return analyser


def _load_face_swapper():
    import insightface

    model_path = Path(settings.swapper_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Swap model not found at '{model_path}'. See README.md for download "
            "instructions for inswapper_128.onnx."
        )
    return insightface.model_zoo.get_model(str(model_path), providers=settings.onnx_providers)


def _load_face_enhancer():
    """Optional GFPGAN restoration pass to sharpen/clean the swapped face."""
    if not settings.enable_face_enhancer:
        return None
    try:
        import torch
        from gfpgan import GFPGANer

        device = "cuda" if settings.use_cuda and torch.cuda.is_available() else "cpu"
        if settings.use_cuda and device == "cpu":
            logger.warning(
                "GFPGAN GPU requested but CUDA is unavailable; falling back to CPU."
            )

        return GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            device=device,
        )
    except Exception as exc:  # pragma: no cover - enhancer is best-effort
        logger.warning("Face enhancer unavailable, continuing without it: %s", exc)
        return None


def get_engine():
    """Thread-safe lazy init so the (slow) model load happens once, on first request."""
    global _face_analyser, _face_swapper, _face_enhancer
    if _face_analyser is None or _face_swapper is None:
        with _lock:
            if _face_analyser is None:
                logger.info("Loading face analyser (buffalo_l)...")
                _face_analyser = _load_face_analyser()
            if _face_swapper is None:
                logger.info("Loading face swapper (inswapper_128)...")
                _face_swapper = _load_face_swapper()
            if _face_enhancer is None and settings.enable_face_enhancer:
                logger.info("Loading face enhancer (GFPGAN)...")
                _face_enhancer = _load_face_enhancer()
    return _face_analyser, _face_swapper, _face_enhancer


def get_primary_face(analyser, image: np.ndarray, want: str = "largest"):
    """Return the most prominent detected face in `image`, or raise NoFaceFoundError."""
    faces = analyser.get(image)
    if not faces:
        raise NoFaceFoundError("No face detected in the supplied image.")
    if want == "largest":
        def area(f):
            x1, y1, x2, y2 = f.bbox
            return (x2 - x1) * (y2 - y1)
        return max(faces, key=area)
    return faces[0]


def get_all_faces(analyser, image: np.ndarray):
    return analyser.get(image)


def swap_face_in_frame(
    frame: np.ndarray,
    source_face,
    face_swapper,
    face_enhancer=None,
    target_face: Optional[object] = None,
) -> np.ndarray:
    """
    Paste `source_face`'s identity onto the face(s) found in `frame`.
    If `target_face` is given, only that detected face is swapped (used for
    video, where we already know which face object we detected this frame).
    Returns the modified frame (frame is also modified in-place by the model).
    """
    result = face_swapper.get(frame, target_face, source_face, paste_back=True)

    if face_enhancer is not None:
        try:
            _, _, result = face_enhancer.enhance(
                result, has_aligned=False, only_center_face=False, paste_back=True
            )
        except Exception as exc:  # pragma: no cover - enhancer is best-effort
            logger.warning("Face enhancement failed on a frame, using raw swap: %s", exc)

    return result
