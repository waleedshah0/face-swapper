"""
Video swap pipeline.

inswapper_128 is a one-shot *image* model, so for video we apply it frame by
frame: decode every frame, run the same image swap used for photos on it,
re-encode. The original audio track is stripped out before processing and
muxed back onto the finished video afterwards (face swapping doesn't touch
audio, so re-encoding it would be wasted work and quality loss).

This is the simplest correct approach. It has a known limitation: each
frame is swapped independently, so on very shaky/low-quality footage you can
occasionally see slight frame-to-frame flicker. Production-grade tools (e.g.
FaceFusion) add a temporal-smoothing pass on top of the same underlying
model — see README.md for notes on extending this.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from app.core.face_engine import (
    get_engine,
    get_all_faces,
    get_primary_face,
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
    progress_cb: ProgressCB = None,
) -> Path:
    analyser, swapper, enhancer = get_engine()

    face_img = cv2.imread(str(face_path))
    if face_img is None:
        raise ValueError(f"Could not read face image: {face_path}")
    source_face = get_primary_face(analyser, face_img)

    cap = cv2.VideoCapture(str(original_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {original_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    logger.info(
        "Starting video swap: %s (%d frames @ %.1ffps, %dx%d, enhancer=%s)",
        original_path.name, total_frames, fps, width, height,
        "ON (slow on CPU)" if enhancer else "OFF",
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
    frames_with_no_face = 0
    start_time = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            target_faces = get_all_faces(analyser, frame)
            if target_faces:
                for target_face in target_faces:
                    frame = swap_face_in_frame(
                        frame, source_face, swapper, enhancer, target_face=target_face
                    )
            else:
                frames_with_no_face += 1  # keep original frame, don't fail the whole video

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

    if frames_with_no_face:
        logger.warning(
            "%d/%d frames had no detectable face and were left unswapped (%s)",
            frames_with_no_face, frame_idx, original_path.name,
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
