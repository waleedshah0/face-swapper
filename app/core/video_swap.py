"""
Video swap pipeline.

inswapper_128 is a one-shot *image* model, so for video we apply it frame by
frame: decode every frame, run the same image swap used for photos on it,
re-encode. The original audio track is stripped out before processing and
muxed back onto the finished video afterwards (face swapping doesn't touch
audio, so re-encoding it would be wasted work and quality loss).

Which face gets swapped when a frame has more than one (e.g. "2 men, 1
woman"): the same rule as image_swap.py — the face matching TargetSource by
identity if one was given, otherwise the first female face. See
"Locking onto a target" below for how that stays consistent across frames.

This is otherwise the simplest correct approach. It has a known limitation:
each frame is swapped independently, so on very shaky/low-quality footage
you can occasionally see slight frame-to-frame flicker. Production-grade
tools (e.g. FaceFusion) add a temporal-smoothing pass on top of the same
underlying model — see README.md for notes on extending this.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from app.config import settings
from app.core.face_engine import (
    TargetFaceNotFoundError,
    get_all_faces,
    get_engine,
    get_face_embedding,
    get_primary_face,
    select_target_face,
    swap_face_in_frame,
)

logger = logging.getLogger("faceswap.video")

ProgressCB = Optional[Callable[[float], None]]

# How often to print progress to the console. There's no per-frame output by
# default, which makes a slow CPU run look identical to a hung one — this
# line is what makes the process's liveness visible while it's working.
LOG_EVERY_N_FRAMES = 10


def _ffmpeg_has_audio(video_path: Path) -> bool:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True, text=True,
    )
    return bool(probe.stdout.strip())


def swap_video(
    original_path: Path,
    face_path: Path,
    output_path: Path,
    target_path: Optional[Path] = None,
    progress_cb: ProgressCB = None,
) -> Path:
    """
    target_path: optional reference photo of the specific person (in
    original_path) whose face should be replaced throughout the video. If
    omitted, the first female face is used instead — see "Locking onto a
    target" below for how that's kept consistent frame-to-frame.
    """
    analyser, swapper, enhancer = get_engine()

    face_img = cv2.imread(str(face_path))
    if face_img is None:
        raise ValueError(f"Could not read face image: {face_path}")
    source_face = get_primary_face(analyser, face_img)

    # --------------------------------------------------------------------- #
    # Locking onto a target.
    #
    # Each frame is analysed independently (see module docstring), so without
    # care "swap the first female face" could flicker between different
    # people frame to frame as they move around. Instead: resolve a single
    # target embedding once, then match against it for every frame.
    #   - target_path given: that embedding is known upfront.
    #   - target_path not given: unresolved until the first frame where a
    #     female face is actually found; her embedding becomes the target
    #     for every subsequent frame, i.e. the same person throughout the
    #     rest of the video, not just "whoever's leftmost female" each time.
    # --------------------------------------------------------------------- #
    target_embedding: Optional[np.ndarray] = None
    if target_path is not None:
        target_img = cv2.imread(str(target_path))
        if target_img is None:
            raise ValueError(f"Could not read target image: {target_path}")
        target_reference_face = get_primary_face(analyser, target_img)
        target_embedding = get_face_embedding(target_reference_face)

    cap = cv2.VideoCapture(str(original_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {original_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    logger.info(
        "Starting video swap: %s (%d frames @ %.1ffps, %dx%d, enhancer=%s, target=%s)",
        original_path.name, total_frames, fps, width, height,
        "ON (slow on CPU)" if enhancer else "OFF",
        "TargetSource" if target_path is not None else "first female face",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent_path = output_path.with_suffix(".silent.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(silent_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(
            f"Could not open video writer for {silent_path} — the 'mp4v' codec "
            "may be unavailable on this system."
        )

    frame_idx = 0
    frames_with_no_target = 0
    target_ever_matched = False
    start_time = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            target_faces = get_all_faces(analyser, frame)
            matched_face = None
            if target_faces:
                matched_face = select_target_face(
                    target_faces, target_embedding, match_threshold=settings.face_match_threshold
                )
                if matched_face is not None and target_embedding is None:
                    # First frame where the default (first female) target
                    # was found — lock onto her identity for every frame
                    # from here on, instead of re-picking independently.
                    target_embedding = get_face_embedding(matched_face)

            if matched_face is not None:
                target_ever_matched = True
                frame = swap_face_in_frame(
                    frame, source_face, swapper, enhancer, target_face=matched_face
                )
            else:
                # Keep the original frame — target not visible in this frame
                # (or not yet found at all), not a reason to fail the video.
                frames_with_no_target += 1

            writer.write(frame)
            frame_idx += 1

            if total_frames:
                pct = min(99.0, 95.0 * frame_idx / total_frames)
            else:
                pct = 0.0  # frame count unknown for this container/codec
            if progress_cb:
                progress_cb(pct)

            if frame_idx % LOG_EVERY_N_FRAMES == 0 or frame_idx == total_frames:
                elapsed = time.monotonic() - start_time
                avg_per_frame = elapsed / frame_idx
                if total_frames:
                    eta = avg_per_frame * (total_frames - frame_idx)
                    logger.info(
                        "  frame %d/%d (%.0f%%) — %.1fs elapsed, ~%.1fs remaining "
                        "(%.2fs/frame)",
                        frame_idx, total_frames, pct, elapsed, eta, avg_per_frame,
                    )
                else:
                    logger.info(
                        "  frame %d (total unknown) — %.1fs elapsed (%.2fs/frame)",
                        frame_idx, elapsed, avg_per_frame,
                    )
    finally:
        cap.release()
        writer.release()

    total_elapsed = time.monotonic() - start_time
    logger.info(
        "Finished swapping %d frames in %.1fs (%.2fs/frame avg)",
        frame_idx, total_elapsed, total_elapsed / frame_idx if frame_idx else 0.0,
    )

    if frames_with_no_target:
        logger.warning(
            "%d/%d frames had no matching target face and were left unswapped (%s)",
            frames_with_no_target, frame_idx, original_path.name,
        )
    if not target_ever_matched:
        # Never found a target at all, in any frame — the whole video went
        # out the door unswapped. That's a failure, not a quiet no-op.
        silent_path.unlink(missing_ok=True)
        if target_path is not None:
            raise TargetFaceNotFoundError(
                "Could not find the person from TargetSource in any frame of the video."
            )
        raise TargetFaceNotFoundError(
            "No female face was detected in any frame of the video to use as the "
            "default swap target. Provide TargetSource to choose a specific person instead."
        )

    logger.info("Muxing original audio onto the swapped video...")
    _mux_audio(original_path, silent_path, output_path)
    silent_path.unlink(missing_ok=True)

    if progress_cb:
        progress_cb(100.0)

    logger.info("Video swap complete: %s", output_path)
    return output_path


def _mux_audio(original_with_audio: Path, swapped_silent: Path, final_out: Path) -> None:
    """Copy the audio track from the original video onto the newly swapped (silent) one."""
    if not _ffmpeg_has_audio(original_with_audio):
        # nothing to mux, just rename the silent version into place
        swapped_silent.replace(final_out)
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", str(swapped_silent),
        "-i", str(original_with_audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(final_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg audio mux failed, falling back to silent video: %s", result.stderr)
        swapped_silent.replace(final_out)
