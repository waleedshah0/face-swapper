"""
Image swap pipeline: load source/face images -> detect faces -> swap -> save.
Synchronous and fast (sub-second to a couple of seconds on a GPU), so the
image endpoint can simply await this and return the result directly.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from app.core.face_engine import (
    NoFaceFoundError,
    get_all_faces,
    get_engine,
    get_primary_face,
    swap_face_in_frame,
)


def swap_image(original_path: Path, face_path: Path, output_path: Path) -> Path:
    """
    original_path: the photo whose face(s) will be replaced
    face_path:     the photo of the face to insert
    output_path:   where to write the resulting image

    Every face detected in `original_path` is swapped with the single face
    found in `face_path`. Raises NoFaceFoundError if either image has no
    detectable face.
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

    result = original_img
    for target_face in target_faces:
        result = swap_face_in_frame(
            result, source_face, swapper, enhancer, target_face=target_face
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    return output_path
