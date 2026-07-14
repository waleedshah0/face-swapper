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
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger("faceswap.engine")

_lock = threading.Lock()
_face_analyser = None
_face_swapper = None
# None = not yet attempted. Once attempted, a list of (name, GFPGANer
# instance) for whichever of ENABLE_GFPGAN/ENABLE_RESTOREFORMER loaded
# successfully — possibly empty if nothing is enabled or everything enabled
# failed to load. See _load_face_enhancers().
_face_enhancers = None
_face_enhancer_disabled_reason = None


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


# Both restoration models below ship inside the already-installed `gfpgan`
# package (GFPGANer just takes a different `arch` + checkpoint), so enabling
# either (or both) needs no new dependency. See ENABLE_GFPGAN /
# ENABLE_RESTOREFORMER in config.py for the tradeoffs. Order here also
# defines chain order when both are enabled — see _load_face_enhancers().
_ENHANCER_MODELS = {
    "gfpgan": {
        "arch": "clean",
        "channel_multiplier": 2,
        "model_path": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    },
    "restoreformer": {
        "arch": "RestoreFormer",
        "channel_multiplier": 2,
        "model_path": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
    },
}


def _patch_torchvision_functional_tensor() -> None:
    """
    basicsr==1.4.2 (a gfpgan dependency) imports
    `torchvision.transforms.functional_tensor`, a module torchvision
    removed in 0.17+ (its contents — e.g. rgb_to_grayscale — moved into
    `torchvision.transforms.functional` under the same names). basicsr is
    unmaintained since 2022 and won't be updated to match, so this
    registers a module alias satisfying that import without patching
    basicsr's source or downgrading torchvision. Must run before
    `from gfpgan import GFPGANer`, which imports basicsr transitively.
    Safe/idempotent to call more than once.
    """
    import sys

    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    import torchvision.transforms.functional as _functional
    sys.modules["torchvision.transforms.functional_tensor"] = _functional


def _check_cuda_enhancer_readiness(torch) -> None:
    """
    Raises if the installed PyTorch wheel doesn't include compiled kernels
    for this GPU's compute capability — better to fail once here, at model
    load, than once per processed frame. Also runs one real kernel so an
    incompatible wheel fails immediately rather than on the first frame.
    """
    major, minor = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{major}{minor}"
    compiled_arches = set(torch.cuda.get_arch_list())
    logger.info(
        "Enhancer PyTorch=%s, CUDA runtime=%s, GPU=%s (%s), compiled_arches=%s",
        torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0),
        required_arch, sorted(compiled_arches),
    )
    if required_arch not in compiled_arches:
        raise RuntimeError(
            f"Installed PyTorch {torch.__version__} does not include {required_arch} kernels "
            f"for {torch.cuda.get_device_name(0)}. Install a CUDA 12.8+ PyTorch wheel "
            "(torch>=2.7, torchvision matching torch) from "
            "https://download.pytorch.org/whl/cu128."
        )
    torch.zeros(1, device="cuda").add_(1)
    torch.cuda.synchronize()


def _load_face_enhancers() -> List[Tuple[str, object]]:
    """
    Load whichever restoration model(s) are enabled (ENABLE_GFPGAN /
    ENABLE_RESTOREFORMER — see config.py for the full truth table), in a
    fixed order (gfpgan, then restoreformer) matching _ENHANCER_MODELS, so
    when both are on, _apply_face_enhancers() always chains them the same
    way: GFPGAN restores first, RestoreFormer refines its output.

    Each model is loaded independently and best-effort: if one fails (e.g. a
    weights download hiccup) the other can still load and run, so a single
    bad model degrades to partial rather than zero enhancement. Returns []
    if nothing is enabled, the `gfpgan` package itself is unavailable, or
    every enabled model failed to load.
    """
    enabled_names = [
        name
        for name, flag in (
            ("gfpgan", settings.enable_gfpgan),
            ("restoreformer", settings.enable_restoreformer),
        )
        if flag
    ]
    if not enabled_names:
        return []

    try:
        import torch

        _patch_torchvision_functional_tensor()
        from gfpgan import GFPGANer
    except Exception as exc:  # pragma: no cover - enhancer is best-effort
        logger.warning("Face enhancer package unavailable, continuing without it: %s", exc)
        return []

    device = "cuda" if settings.use_cuda and torch.cuda.is_available() else "cpu"
    if settings.use_cuda and device == "cpu":
        logger.warning(
            "Face enhancer GPU requested but CUDA is unavailable; falling back to CPU."
        )

    if device == "cuda":
        try:
            _check_cuda_enhancer_readiness(torch)
        except Exception as exc:
            logger.warning("Face enhancer unavailable, continuing without it: %s", exc)
            return []

    loaded: List[Tuple[str, object]] = []
    for name in enabled_names:
        model_config = _ENHANCER_MODELS[name]
        try:
            enhancer = GFPGANer(
                model_path=model_config["model_path"],
                upscale=1,
                arch=model_config["arch"],
                channel_multiplier=model_config["channel_multiplier"],
                device=device,
            )
            loaded.append((name, enhancer))
            logger.info("Loaded face enhancer: %s (device=%s)", name, device)
        except Exception as exc:  # pragma: no cover - enhancer is best-effort
            logger.warning("Face enhancer '%s' unavailable, continuing without it: %s", name, exc)

    return loaded


def get_engine():
    """Thread-safe lazy init so the (slow) model load happens once, on first request."""
    global _face_analyser, _face_swapper, _face_enhancers, _face_enhancer_disabled_reason
    needs_enhancer = (
        settings.enable_face_enhancer
        and _face_enhancers is None
        and _face_enhancer_disabled_reason is None
    )
    if _face_analyser is None or _face_swapper is None or needs_enhancer:
        with _lock:
            if _face_analyser is None:
                logger.info("Loading face analyser (buffalo_l)...")
                _face_analyser = _load_face_analyser()
            if _face_swapper is None:
                logger.info("Loading face swapper (inswapper_128)...")
                _face_swapper = _load_face_swapper()
            if (
                settings.enable_face_enhancer
                and _face_enhancers is None
                and _face_enhancer_disabled_reason is None
            ):
                enabled_names = [
                    name
                    for name, flag in (
                        ("gfpgan", settings.enable_gfpgan),
                        ("restoreformer", settings.enable_restoreformer),
                    )
                    if flag
                ]
                logger.info("Loading face enhancer(s): %s...", ", ".join(enabled_names))
                _face_enhancers = _load_face_enhancers()
                if not _face_enhancers:
                    _face_enhancer_disabled_reason = "enhancer initialization failed"
    return _face_analyser, _face_swapper, _face_enhancers or []


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


# --------------------------------------------------------------------------- #
# Color correction.
#
# inswapper_128 pastes the source identity's face as-is — it doesn't correct
# for skin-tone/lighting differences between the SwapSource photo and the
# frame it's pasted into, which is a big part of why raw swaps can look
# "pasted on". This shifts the pasted face's color statistics (in LAB space,
# which separates lightness from color) to match the frame it landed in,
# using the pre-swap frame as the lighting reference, then blends the
# correction back with a feathered mask so the fix itself doesn't introduce
# a new hard edge. Cheap: numpy/cv2 only, no extra model, negligible cost
# next to the swap model itself. Toggle via ENABLE_COLOR_CORRECTION.
# --------------------------------------------------------------------------- #

_COLOR_CORRECTION_PADDING_RATIO = 0.15  # extra context sampled around the face box


def _match_color_lab(source_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
    """Shift source_bgr's color/lighting statistics to match reference_bgr."""
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    source_mean, source_std = source_lab.mean(axis=(0, 1)), source_lab.std(axis=(0, 1))
    reference_mean, reference_std = reference_lab.mean(axis=(0, 1)), reference_lab.std(axis=(0, 1))
    source_std = np.clip(source_std, 1e-3, None)

    corrected = (source_lab - source_mean) * (reference_std / source_std) + reference_mean
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


def _feathered_box_mask(height: int, width: int, box: tuple) -> np.ndarray:
    """A float32 HxWx1 mask: 1.0 inside `box`, fading to 0.0 over a soft edge."""
    x1, y1, x2, y2 = box
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, thickness=-1)
    feather = max(5, int(0.2 * min(x2 - x1, y2 - y1))) | 1  # odd kernel size required
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    return mask[:, :, None]


def _color_correct_pasted_face(
    swapped_frame: np.ndarray, pre_swap_frame: np.ndarray, bbox
) -> np.ndarray:
    """
    Correct the just-pasted face region in `swapped_frame` so its color/
    lighting matches the frame, using `pre_swap_frame` (the same frame
    before swapping) as the reference for what that lighting looks like.
    """
    h, w = swapped_frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return swapped_frame

    pad_x = int((x2 - x1) * _COLOR_CORRECTION_PADDING_RATIO)
    pad_y = int((y2 - y1) * _COLOR_CORRECTION_PADDING_RATIO)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

    fake_crop = swapped_frame[cy1:cy2, cx1:cx2]
    reference_crop = pre_swap_frame[cy1:cy2, cx1:cx2]
    if fake_crop.size == 0 or reference_crop.size == 0:
        return swapped_frame

    corrected_crop = _match_color_lab(fake_crop, reference_crop)
    mask = _feathered_box_mask(
        cy2 - cy1, cx2 - cx1, (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
    )
    blended_crop = (
        corrected_crop.astype(np.float32) * mask + fake_crop.astype(np.float32) * (1 - mask)
    ).astype(np.uint8)

    output = swapped_frame.copy()
    output[cy1:cy2, cx1:cx2] = blended_crop
    return output


# --------------------------------------------------------------------------- #
# Eyewear protection.
#
# GFPGAN's restoration model is trained mostly on bare faces, and glasses
# are a common failure case: lens glare/reflections and frame edges get
# misread as noise/artifacts and "corrected" away, which can warp or blur
# the glasses. There's no separate glasses-detection model here — instead
# this reuses the 5-point landmarks InsightFace already computed (left eye,
# right eye, ...) to build a band over the eye/glasses area, and blends
# GFPGAN's output back toward the pre-enhancement (post-swap) frame in that
# band, so the rest of the face still gets sharpened normally. Toggle via
# PROTECT_EYEWEAR_REGION; tune how strong the protection is via
# EYEWEAR_PROTECTION_STRENGTH.
# --------------------------------------------------------------------------- #


def _eye_band_mask(height: int, width: int, kps) -> Optional[np.ndarray]:
    """
    A float32 HxWx1 mask, 1.0 over the eye/glasses band (fading out via a
    Gaussian blur), built from `kps` — the face's 5-point landmarks
    (left eye, right eye, nose, mouth corners), in that order, as returned
    by InsightFace. Returns None if landmarks aren't usable.
    """
    if kps is None or len(kps) < 2:
        return None

    left_eye = np.asarray(kps[0], dtype=np.float32)
    right_eye = np.asarray(kps[1], dtype=np.float32)
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist <= 0:
        return None

    cx, cy = (left_eye + right_eye) / 2.0
    half_w = eye_dist * 1.3
    half_h = eye_dist * 0.65

    x1, y1 = int(cx - half_w), int(cy - half_h)
    x2, y2 = int(cx + half_w), int(cy + half_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, thickness=-1)
    feather = max(5, int(0.6 * (y2 - y1))) | 1  # odd kernel size required
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    return mask[:, :, None]


def _apply_face_enhancers(
    result: np.ndarray,
    face_enhancers: List[Tuple[str, object]],
    target_face: Optional[object],
) -> np.ndarray:
    """
    Run each enabled restoration model in turn — see ENABLE_GFPGAN /
    ENABLE_RESTOREFORMER in config.py. `face_enhancers` is already in chain
    order (gfpgan, then restoreformer — see _load_face_enhancers()), so when
    both are enabled, RestoreFormer refines GFPGAN's output rather than the
    other way around. With only one enabled, this is just that one pass.

    Eye/glasses-band protection (if enabled) is applied once, at the end,
    relative to the frame as it looked before ANY enhancer ran — not
    per-stage — so the protection strength doesn't compound when chaining
    two models.

    Note: `weight` only affects GFPGAN's "clean" arch (it blends between
    restored and original in an intermediate style layer specific to that
    architecture). RestoreFormer's forward() accepts and ignores it via
    **kwargs, so FACE_ENHANCER_WEIGHT only has an effect on the GFPGAN stage.
    """
    pre_enhance = result.copy()
    current = result
    for _name, enhancer in face_enhancers:
        _, _, current = enhancer.enhance(
            current, has_aligned=False, only_center_face=False, paste_back=True,
            weight=settings.face_enhancer_weight,
        )

    if not settings.protect_eyewear_region or target_face is None:
        return current

    kps = getattr(target_face, "kps", None)
    mask = _eye_band_mask(current.shape[0], current.shape[1], kps)
    if mask is None:
        return current

    protect = mask * settings.eyewear_protection_strength
    blended = (
        current.astype(np.float32) * (1 - protect) + pre_enhance.astype(np.float32) * protect
    ).astype(np.uint8)
    return blended


def swap_face_in_frame(
    frame: np.ndarray,
    source_face,
    face_swapper,
    face_enhancers: Optional[List[Tuple[str, object]]] = None,
    target_face: Optional[object] = None,
) -> np.ndarray:
    """
    Paste `source_face`'s identity onto `target_face` in `frame`. Returns the
    modified frame.

    Two optional, cheap post-processing passes run after the raw model swap:
      - color correction (ENABLE_COLOR_CORRECTION, default on): fixes the
        "pasted on" look by matching the swapped face's lighting/skin tone
        to the frame. See _color_correct_pasted_face() above.
      - face enhancer(s) (ENABLE_GFPGAN / ENABLE_RESTOREFORMER, both off by
        default): one or both restoration passes, chained in that order when
        both are on, with restoration strength tunable via
        FACE_ENHANCER_WEIGHT (GFPGAN stage only), and the eye/glasses band
        optionally protected from known glasses artifacts via
        PROTECT_EYEWEAR_REGION / EYEWEAR_PROTECTION_STRENGTH. See
        _apply_face_enhancers() above.
    Both passes are best-effort — if either fails on a given frame, the swap
    still returns rather than crashing the whole job over a cosmetic pass.
    """
    # face_swapper.get() modifies `frame` in place, so grab a copy first to
    # use as the "what did this area look like before swapping" reference.
    pre_swap_frame = frame.copy()
    result = face_swapper.get(frame, target_face, source_face, paste_back=True)

    if settings.enable_color_correction and target_face is not None:
        try:
            result = _color_correct_pasted_face(result, pre_swap_frame, target_face.bbox)
        except Exception as exc:  # pragma: no cover - correction is best-effort
            logger.warning("Color correction failed on a frame, using raw swap: %s", exc)

    if face_enhancers:
        try:
            result = _apply_face_enhancers(result, face_enhancers, target_face)
        except Exception as exc:  # pragma: no cover - enhancer is best-effort
            global _face_enhancers, _face_enhancer_disabled_reason
            logger.exception(
                "Face enhancement failed; disabling enhancer(s) for this worker process "
                "and using raw swaps for remaining frames: %s", exc
            )
            _face_enhancers = []
            _face_enhancer_disabled_reason = str(exc)

    return result
