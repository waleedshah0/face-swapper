<<<<<<< HEAD
# Face Swap Studio

A self-contained website + FastAPI backend for swapping faces in **images**
and **videos**, built around InsightFace's `inswapper_128` model (the
one-shot face-swap model used under the hood by most production face-swap
tools, including the open-source FaceFusion project).

```
Browser  ──────►  FastAPI (app/main.py)
  upload                │
  original + face        ├─ /api/swap/image   → sync, returns the swapped image
                         └─ /api/swap/video   → starts a background job, poll for status
                                                      │
                                              app/core/face_engine.py
                                              (InsightFace detector + inswapper_128)
                                                      │
                                              app/core/image_swap.py
                                              app/core/video_swap.py (frame loop + ffmpeg audio mux)
```

This is a single-process reference implementation. If you want to split it
into the 4-step architecture you described earlier (web server / GPU server
talking through a shared folder), `app/core/*` is the part that runs on the
GPU box — `image_swap.swap_image()` and `video_swap.swap_video()` are the
two functions a GPU-side worker would call after picking files up from the
shared folder, before writing the result back and notifying the web server.

## 1. Install system dependencies

- Python 3.10+
- **ffmpeg** on your PATH (`apt install ffmpeg` / `brew install ffmpeg` / on Windows, [download a build](https://www.gyan.dev/ffmpeg/builds/) and add it to PATH) — required for video audio muxing

This project runs CPU-only by default — no GPU required to get started.

## 2. Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you get a GPU later: change `onnxruntime` to `onnxruntime-gpu` in
`requirements.txt`, set `EXECUTION_PROVIDER=cuda` in `.env`, and use
`Dockerfile.gpu` instead of `Dockerfile`.

### What to expect on CPU

- **Images**: a few seconds each — perfectly usable.
- **Video**: this is the part that's genuinely slow on CPU, because every
  single frame runs a full detection + swap pass. Roughly 1-3 seconds per
  frame is typical on a modern laptop CPU, so a 10-second clip at 30fps
  (300 frames) can take 5-15 minutes. Plan accordingly:
  - Test with short clips (a few seconds) first.
  - `ENABLE_FACE_ENHANCER` defaults to `false` here on purpose — GFPGAN
    roughly doubles per-frame time on CPU for a quality bump you often
    won't need for testing.
  - `MAX_VIDEO_MB` already defaults to 50MB in `.env.example` for this
    reason — drop it further if you want a tighter safety margin.
  - If video volume becomes real (not just testing), that's the point to
    rent GPU time (a cloud GPU instance, or a local card) rather than scale
    CPU workers — a single mid-range GPU will outrun a large CPU fleet for
    this workload.

## 3. Get the model weights

`inswapper_128.onnx` is distributed by the InsightFace project directly
(not bundled here, and not on PyPI). Search "inswapper_128.onnx download"
for the current official InsightFace model-zoo / Hugging Face mirror link,
verify the file checksum against InsightFace's own documentation, and place
it at:

```
faceswap-app/models/inswapper_128.onnx
```

The face *detection* model (`buffalo_l`) downloads automatically the first
time you run the app (InsightFace fetches it to `~/.insightface/models/`).

If you want the optional face enhancer, install its dependencies first
(`pip install -r requirements-enhancer.txt` — see the troubleshooting note
below if that fails to build) and set `ENABLE_FACE_ENHANCER=true`. The
GFPGAN weights then download automatically on first run.

## 4. Configure

```bash
cp .env.example .env
# edit .env: execution provider, model path, upload size limits, etc.
```

## 5. Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` — you'll see two tabs, **Image** and **Video**.
Each lets you upload an original + a face photo and run the swap.

## API reference

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/api/swap/image` | multipart: `original_image`, `face_image`, `consent` | Returns the swapped PNG directly |
| POST | `/api/swap/video` | multipart: `original_video`, `face_image`, `consent` | Returns `{job_id, status}` immediately |
| GET | `/api/jobs/{job_id}` | — | Returns `{status, progress, download_url}` |
| GET | `/api/jobs/{job_id}/download` | — | Streams the finished MP4 |

Why video is async and image isn't: image swap is a single inference call
(sub-second to a couple seconds on GPU). Video is hundreds/thousands of
per-frame inference calls plus an ffmpeg mux step, so it runs as a
background task and the frontend polls for progress instead of blocking
the HTTP request.

## Performance / scaling notes

- **Model loading is lazy and cached** (`app/core/face_engine.py`) — the
  first request after server start pays the model-load cost, every request
  after that reuses the already-loaded model in memory.
- **Video is the bottleneck.** Each frame runs through detection + swap.
  For long videos, consider: processing on a frame-skip + interpolation
  schedule, running multiple GPU workers behind a queue (Redis/RQ or
  Celery) instead of in-process `BackgroundTasks`, or capping max video
  length/resolution on upload.
- The in-memory job store (`app/jobs.py`) is fine for one process. For
  multiple workers or servers, replace it with Redis or a database table —
  the `create_job/get_job/update_job` interface is intentionally tiny so
  that's a drop-in swap.
- `inswapper_128` is a **one-shot** model — no per-face-pair training step,
  which is what makes it suitable for an on-demand public-facing service
  (compare to DeepFaceLab, which needs hours of training per face pair and
  fits offline/VFX workflows better than live web traffic).

## Responsible use

- The consent checkbox in both forms is enforced server-side
  (`_require_consent` in `app/main.py`) — requests without it are rejected
  with a 400.
- Consider adding, depending on your jurisdiction and audience: visible
  watermarking of outputs, content moderation on uploads, rate limiting,
  and logging/audit trails. Non-consensual use of this kind of tool carries
  real legal exposure in a growing number of jurisdictions (e.g. the U.S.
  TAKE IT DOWN Act) — worth a compliance review before launch, not after.

## Troubleshooting

**`pip` can't find `onnxruntime-gpu==1.18.0` / `onnxruntime==1.18.0`**
PyPI stops publishing wheels for old patch versions once they no longer
build against current Python releases. `requirements.txt` now uses
`onnxruntime-gpu>=1.19.0` (a floor, not an exact pin) so pip picks the
newest build that exists for your Python version. If you hit the same
"no matching distribution" error on any other package in this file, the
fix is the same: relax `==` to `>=` for that line.

**`basicsr` fails to build with `KeyError: '__version__'`**
This happens if you install from `requirements-enhancer.txt` (the optional
face enhancer) on Python 3.12+ — `basicsr==1.4.2`'s `setup.py` uses an old
version-parsing trick that breaks on newer Python and the package hasn't
been updated since 2022. It's not your environment. Two ways out:
1. Skip the enhancer — the app runs fine without it (it's off by default).
2. If you do want it, create the venv with Python 3.10 or 3.11 specifically
   for this project (`py -3.11 -m venv .venv` on Windows, or
   `python3.11 -m venv .venv` elsewhere) — `basicsr`'s old setup.py still
   works there.

**GPU not picked up / `onnxruntime-gpu` silently falls back to CPU**
`onnxruntime-gpu` needs a CUDA + cuDNN version on your system that matches
the onnxruntime release you installed (check the [onnxruntime CUDA
compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
for the version you installed). Run `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`
— if `CUDAExecutionProvider` isn't in the list, it's a CUDA/driver mismatch,
not an app bug.

**`gfpgan`/`basicsr` import error mentioning `torchvision.transforms.functional_tensor`**
`basicsr==1.4.2` (a GFPGAN dependency) was written against an older
torchvision and breaks on torchvision releases that removed that module.
This only affects the *optional* face enhancer — `face_engine.py` already
catches this and logs a warning instead of crashing, so the app keeps
working without enhancement. To actually fix it: either install an older
`torchvision` (`pip install "torchvision<0.17"`) in the same environment,
or set `ENABLE_FACE_ENHANCER=false` in `.env` and skip it entirely.

**Model load is slow on every request, not just the first one**
That means something is re-creating the `FaceAnalysis`/swapper objects
instead of reusing the cached ones in `face_engine.py` — check you're
running a single `uvicorn` worker (`--workers 1`) during testing, since
each worker process gets its own copy of the cache.

## Extending this

- **Temporal smoothing for video**: the current implementation swaps each
  frame independently, which can flicker slightly on shaky/low-quality
  footage. A face-tracking pass (carry the previous frame's bounding box
  forward instead of re-detecting from scratch every frame) is the next
  upgrade if you see this in practice.
- **Multi-face control**: `image_swap.py`/`video_swap.py` currently swap
  every detected face with the one supplied face. If you need "swap only
  the 2nd person from the left," that's a matter of letting the user pick
  which detected face/bbox to target instead of looping over all of them.
- **Queueing**: swap `BackgroundTasks` for Celery/RQ + Redis once you need
  more than one video worker.
=======
