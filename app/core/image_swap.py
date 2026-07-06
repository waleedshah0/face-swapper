"""
Image swap pipeline: load source/face images -> detect faces -> pick ONE
target face -> swap -> save. Synchronous and fast (sub-second to a couple of
seconds on a GPU), so the image endpoint can simply await this and return
the result directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from app.config import settings
from app.core.face_engine import (
    NoFaceFoundError,
    TargetFaceNotFoundError,
    get_all_faces,
    get_engine,
    get_face_embedding,
    get_primary_face,
    select_target_face,
    swap_face_in_frame,
)


def swap_image(
    original_path: Path,
    face_path: Path,
    output_path: Path,
    target_path: Optional[Path] = None,
) -> Path:
    """
    original_path: the photo whose face will be replaced
    face_path:     the photo of the face to insert
    output_path:   where to write the resulting image
    target_path:   optional reference photo of the specific person (in
                   `original_path`) whose face should be replaced. If a
                   photo has more than one face, this decides which one.

    Which face gets swapped:
      - target_path given: whichever detected face in `original_path` best
        matches the face in `target_path` (by face-recognition embedding,
        not position). Raises TargetFaceNotFoundError if no detected face
        matches closely enough.
      - target_path not given: the first female face, reading left to
        right. Raises TargetFaceNotFoundError if no female face is found.

    Raises NoFaceFoundError if `original_path` or `face_path` has no
    detectable face at all.
    """
    analyser, swapper, enhancer = get_engine()

    original_img = cv2.imread(str(original_path))
    face_img = cv2.imread(str(face_path))

    if original_img is None:
        raise ValueError(f"Could not read original image: {original_path}")
    if face_img is None:
        raise ValueError(f"Could not read face image: {face_path}")

    source_face = get_primary_face(analyser, face_img)

    target_faces = get_all_faces(analyser, original_img)
    if not target_faces:
        raise NoFaceFoundError("No face detected in the original image.")

    target_embedding = None
    if target_path is not None:
        target_img = cv2.imread(str(target_path))
        if target_img is None:
            raise ValueError(f"Could not read target image: {target_path}")
        target_reference_face = get_primary_face(analyser, target_img)
        target_embedding = get_face_embedding(target_reference_face)

    matched_face = select_target_face(
        target_faces, target_embedding, match_threshold=settings.face_match_threshold
    )
    if matched_face is None:
        if target_embedding is not None:
            raise TargetFaceNotFoundError(
                "Could not find the person from TargetSource in the original image."
            )
        raise TargetFaceNotFoundError(
            "No female face was detected in the original image to use as the default "
            "swap target. Provide TargetSource to choose a specific face instead."
        )

    result = swap_face_in_frame(
        original_img, source_face, swapper, enhancer, target_face=matched_face
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    return output_path
