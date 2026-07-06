"""
Loads the face detection/analysis model and the face-swap model exactly once
per process, and exposes simple helper functions on top of them.

Models used:
  - buffalo_l (InsightFace FaceAnalysis): face detection, landmarks, embeddings,
    and gender/age (used below to pick a default target face when none is
    specified explicitly)
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
from typing import List, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger("faceswap.engine")

_lock = threading.Lock()
_face_analyser = None
_face_swapper = None
_face_enhancer = None


class NoFaceFoundError(Exception):
    """Raised when no face at all can be detected in a supplied image."""


class TargetFaceNotFoundError(NoFaceFoundError):
    """
    Raised when face(s) were detected, but none of them could be resolved as
    *the* face to swap: either no detected face matched TargetSource closely
    enough, or (when TargetSource wasn't given) no female face was found to
    use as the default target. Subclasses NoFaceFoundError so existing
    `except NoFaceFoundError` handlers keep working unchanged.
    """


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


# --------------------------------------------------------------------------- #
# Choosing *which* detected face to swap.
#
# A photo or video frame can have several faces in it (e.g. "2 men, 1 woman").
# swap_image()/swap_video() used to swap every detected face — that's wrong
# once there's more than one person in frame. The rule now is:
#
#   - If a TargetSource reference photo was supplied: swap whichever detected
#     face's embedding most closely matches the face in that photo (i.e. the
#     same person, found by face recognition, not by position).
#   - Otherwise: swap the first female face, reading left-to-right (leftmost
#     bounding box first). If nothing in the frame is recognised as female,
#     no default target exists and TargetFaceNotFoundError is raised.
# --------------------------------------------------------------------------- #

# Cosine similarity between two face embeddings from the SAME identity is
# typically well above this on buffalo_l; different people are typically
# well below it. Tunable via FACE_MATCH_THRESHOLD if it needs adjusting for
# your footage (harder lighting/angles push same-person scores down).
DEFAULT_FACE_MATCH_THRESHOLD = 0.35


def get_face_embedding(face) -> np.ndarray:
    return np.asarray(face.embedding, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def is_female(face) -> bool:
    """
    InsightFace's genderage model (bundled in buffalo_l) reports gender as
    0 = female, 1 = male on each detected Face object. Defensive default of
    False if gender wasn't computed for some reason (e.g. a stripped model
    pack), so an unknown-gender face is simply not picked as the default
    target rather than crashing.
    """
    gender = getattr(face, "gender", None)
    return gender == 0


def select_target_face(
    faces: List[object],
    target_embedding: Optional[np.ndarray] = None,
    match_threshold: float = DEFAULT_FACE_MATCH_THRESHOLD,
):
    """
    Pick the one face (out of `faces`, all detected in the same image/frame)
    to actually swap.

    - target_embedding given: return whichever face is the closest match by
      cosine similarity, or None if the best match is still below
      match_threshold (i.e. that person isn't actually in this image/frame).
    - target_embedding is None: return the leftmost face recognised as
      female, or None if there isn't one.
    """
    if not faces:
        return None

    if target_embedding is not None:
        best_face, best_score = None, -1.0
        for face in faces:
            score = cosine_similarity(get_face_embedding(face), target_embedding)
            if score > best_score:
                best_face, best_score = face, score
        if best_face is not None and best_score >= match_threshold:
            return best_face
        return None

    faces_left_to_right = sorted(faces, key=lambda f: f.bbox[0])
    for face in faces_left_to_right:
        if is_female(face):
            return face
    return None


def swap_face_in_frame(
    frame: np.ndarray,
    source_face,
    face_swapper,
    face_enhancer=None,
    target_face: Optional[object] = None,
) -> np.ndarray:
    """
    Paste `source_face`'s identity onto `target_face` in `frame`. Returns the
    modified frame (frame is also modified in-place by the model).
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
