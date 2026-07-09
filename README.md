# Face Swap Studio

A backend-only FastAPI service for swapping faces in **images** and
**videos**, built around InsightFace's `inswapper_128` model (the one-shot
face-swap model used under the hood by most production face-swap tools,
including the open-source FaceFusion project).

There is no frontend here — the caller is another backend system (a
website/mobile server) that has already dropped the source files into a
shared uploads folder and just needs the swap done.

```
Caller (website/mobile server)
   │  POST /api/swap  (XML: OriginalSource, SwapSource, optional TargetSource)
   ▼
FastAPI (app/main.py)
   │  validate payload → generate job_id → write status "Starting" to Redis → publish job to RabbitMQ
   │  responds 202 immediately with {job_id, media_type, status}
   ▼
RabbitMQ (swap_jobs queue)
   ▼
Worker process(es) (app/worker.py) — one or more, scale horizontally
   │  status → "In progress" (message updated with rough % complete for video)
   │  app/core/face_engine.py   (InsightFace detector + inswapper_128)
   │  app/core/image_swap.py / app/core/video_swap.py (frame loop + ffmpeg audio mux)
   │  status → "Completed" (+ output_file) or "Failed" (+ message)
   ▼
Redis (status cache, keyed by job_id)
   ▲
   │  GET /api/swap/{job_id}  (one-shot snapshot — call again to see the next update)
Caller
```

Why a queue at all: a single swap can take anywhere from under a second
(image, GPU) to several minutes (video, CPU). Handling requests inline would
mean one slow video ties up a request thread while every other caller waits.
RabbitMQ decouples "accept the request" from "do the work," and lets you run
as many worker processes as you have CPU/GPU capacity for, independently of
how many API requests are coming in.

Note the API container never touches the GPU: `app/main.py` only validates
requests and talks to RabbitMQ/Redis, it never imports the swap pipeline.
Only `app/worker.py` needs GPU access — see "Deploying to a server" below.

## 1. Install system dependencies

- Python 3.10+
- **ffmpeg** on your PATH (`apt install ffmpeg` / `brew install ffmpeg` / on
  Windows, [download a build](https://www.gyan.dev/ffmpeg/builds/) and add
  it to PATH) — required for video audio muxing
- **RabbitMQ** and **Redis** — either run them yourself, or use the included
  `docker-compose.yml` (see step 5)

This project runs CPU-only by default — no GPU required to get started.

## 2. Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you get a GPU later: change `onnxruntime` to `onnxruntime-gpu` in
`requirements.txt`, set `EXECUTION_PROVIDER=cuda` in `.env`, and use
`Dockerfile.gpu` instead of `Dockerfile` (or see "Deploying to a server").

If you want to run everything on GPU, also install the optional enhancer
requirements with `pip install -r requirements-enhancer.txt` and keep
`ENABLE_FACE_ENHANCER=true` in `.env`.

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
    this workload, and run more `app/worker.py` processes in the meantime.

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
`buffalo_l` also includes the gender/age model used to pick a default swap
target — see "Which face gets swapped" below.

If you want the optional face enhancer, install its dependencies first
(`pip install -r requirements-enhancer.txt` — see the troubleshooting note
below if that fails to build) and set `ENABLE_FACE_ENHANCER=true`. The
GFPGAN weights then download automatically on first run.

## 4. Configure

```bash
cp .env.example .env
# edit .env: execution provider, model path, upload size limits,
# RABBITMQ_URL, REDIS_URL, etc.
```

`RABBITMQ_URL` and `REDIS_URL` default to `localhost` — correct if you're
running RabbitMQ/Redis directly on your machine. `docker-compose.yml`
overrides both to point at its own `rabbitmq`/`redis` service containers, so
you don't need to touch `.env` if you're using it.

## 5. Run

**Option A — docker-compose (recommended, brings up everything):**

```bash
docker compose up --build
# scale workers for more swap throughput, e.g.:
docker compose up --build --scale worker=3
```

This starts RabbitMQ (with its management UI at `http://localhost:15672`,
guest/guest), Redis, the API (`http://localhost:8000`), and one worker.

**Option B — run each piece yourself:**

```bash
# terminal 1: RabbitMQ + Redis need to be running and reachable at RABBITMQ_URL / REDIS_URL
# terminal 2:
docker compose up -d rabbitmq redis
uvicorn app.main:app --reload --port 8000
# terminal 3 (one or more):
python -m app.worker
```

docker compose stop rabbitmq redis
docker compose rm -f rabbitmq redis
## Deploying to a server (GPU)

The image is built once and used for **both** the API and the worker — the
worker's `command:` is just overridden to run `python -m app.worker`
instead of `uvicorn`. Only the worker needs `--gpus`.

**1. Build and push (same as before):**

```bash
docker build -t dev1shayansolutions/faceswapper:v1 .
docker push dev1shayansolutions/faceswapper:v1
```

**2. On the server**, create a directory (e.g. `/opt/faceswap-app/`) with
two files — you don't need to check out the repo on the server at all:

- `docker-compose.prod.yml` (copy from this repo)
- `.env`, based on `.env.example` plus:

  ```bash
  EXECUTION_PROVIDER=cuda
  UPLOADS_DIR=/mnt/gpu/send_to_gpu
  OUTPUTS_DIR=/mnt/gpu/receive_from_gpu
  # RABBITMQ_URL / REDIS_URL are overridden by docker-compose.prod.yml
  # to point at the rabbitmq/redis services — no need to set them here.
  ```

  `.env` is **not** baked into the image (see `.dockerignore`) — it's read
  by each container at start time via `env_file:` in
  `docker-compose.prod.yml`, so changing settings later is just an edit +
  restart, not a rebuild.

**3. Pull and start everything:**

```bash
docker pull dev1shayansolutions/faceswapper:v1
docker compose -f docker-compose.prod.yml up -d
# more worker throughput:
docker compose -f docker-compose.prod.yml up -d --scale worker=3
```

This replaces the old single-container `docker run --gpus all ...` command
— `docker-compose.prod.yml` starts RabbitMQ, Redis, the API (on `:8000`,
no GPU), and the worker (with `--gpus`) together, wired to talk to each
other, with the same `/mnt/gpu/send_to_gpu` / `/mnt/gpu/receive_from_gpu`
bind mounts your old command used.

**Rolling out a new version:** build + push a new tag as before, then on
the server:

```bash
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

**Security note:** `docker-compose.prod.yml` binds RabbitMQ's and Redis's
ports to `127.0.0.1` only (containers still reach each other over the
compose network by service name) — only the API's `:8000` is open
externally. If you need the RabbitMQ management UI from outside the
server, tunnel in over SSH rather than exposing `15672` publicly, and
consider changing the default `guest`/`guest` credentials.

## API reference

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/api/swap` | `application/xml`: `<SwapRequest><OriginalSource/><SwapSource/><TargetSource/></SwapRequest>` | Validates the payload, generates a `job_id`, queues the job, returns `202` with `{job_id, media_type, status}` immediately. Does not wait for the swap to finish. |
| GET | `/api/swap/{job_id}` | — | Returns the job's current status as a single JSON response and closes immediately. Not a stream — poll it again whenever you want the next update. |

`OriginalSource`, `SwapSource`, and `TargetSource` are filenames expected to
already exist in `settings.uploads_dir` (the shared folder the
website/mobile server drops files into). Whether this is an image-on-image
or image-on-video swap is decided purely by the extension of
`OriginalSource`.

`job_id` is generated server-side in the `POST /api/swap` response — it's
the *only* identifier for a job, there's no caller-supplied id in the
request. Hang on to the `job_id` you get back to check status later. Because
there's no caller-supplied id to de-duplicate against, every `POST
/api/swap` call queues a brand new job, including if you resend the same
`OriginalSource`/`SwapSource` — retry logic on the caller's side needs to
account for that (e.g. don't blindly retry on timeout without checking
whether the first request actually landed).

### Which face gets swapped (multi-face photos and videos)

`OriginalSource` can have more than one person in it (e.g. "2 men, 1
woman"). `TargetSource` (optional) controls which one actually gets
swapped:

- **`TargetSource` provided** — a reference photo of one specific person.
  Every detected face in `OriginalSource` is compared against the face in
  `TargetSource` by face-recognition embedding (not by position), and
  whichever one matches gets swapped. Everyone else in the shot is left
  alone. If nobody in `OriginalSource` matches closely enough, the job
  fails with a clear message rather than silently swapping the wrong
  person.
- **`TargetSource` omitted** — the **first female face**, reading left to
  right, is swapped by default. If no female face is detected at all, the
  job fails (there's no default to fall back to further).

For video, the same rule applies per frame, but the specific person is
**locked on** once found — either from `TargetSource` up front, or from
whichever frame first contains a female face when `TargetSource` isn't
given — and every later frame matches against that same person's face,
rather than re-picking independently frame to frame (which could otherwise
flicker between different people as they move in and out of shot).

The match is governed by `FACE_MATCH_THRESHOLD` in `.env` (default `0.35`,
cosine similarity) — raise it if the wrong face is occasionally getting
picked, lower it if the right person is being missed.

Job status is one of `Starting` (queued, not yet picked up), `In progress`
(a worker is running the swap — for video jobs, `message` carries a rough
`"NN% complete"` that updates roughly every 5%), `Completed` (`output_file`
is set — the result is in `settings.outputs_dir`), or `Failed` (`message`
explains why — including "no matching/female face found").

Example (`curl`):

```bash
# default target (first female face):
curl -X POST http://localhost:8000/api/swap \
  -H "Content-Type: application/xml" \
  -d '<SwapRequest><OriginalSource>video1.mp4</OriginalSource><SwapSource>image2.jpg</SwapSource></SwapRequest>'

# specific person via TargetSource:
curl -X POST http://localhost:8000/api/swap \
  -H "Content-Type: application/xml" \
  -d '<SwapRequest><OriginalSource>video1.mp4</OriginalSource><SwapSource>image2.jpg</SwapSource><TargetSource>dipika.jpg</TargetSource></SwapRequest>'
# -> {"job_id": "...", "media_type": "video", "status": "Starting"}

curl http://localhost:8000/api/swap/<job_id-from-above>
```

Call the `GET` again a few seconds later to see the next update — this is
plain request/response, not a held-open connection, so a standard HTTP
client (or Swagger's "Try it out") works fine, unlike a Server-Sent Events
or WebSocket stream would.

### Improving swap quality

Two independent, best-effort post-processing passes run after the raw
`inswapper_128` swap, both in `app/core/face_engine.py:swap_face_in_frame()`:

- **Color correction** (`ENABLE_COLOR_CORRECTION`, default `true`) — the raw
  model pastes the source face's own color/lighting as-is, which is most of
  why untouched swaps can look "pasted on". This shifts the pasted face's
  color statistics (in LAB space) to match the frame it landed in, blended
  back with a feathered mask so the fix doesn't add its own hard edge. No
  extra model, negligible cost — leave this on.
- **Face enhancer** (`ENABLE_FACE_ENHANCER`, default `false`) — a GAN-based
  restoration pass that sharpens/cleans the swapped face. Meaningfully
  better output, but roughly doubles per-frame time on CPU — worth it once
  you're on GPU (see "Deploying to a server" above), off by default for
  fast local testing. `FACE_ENHANCER_MODEL` picks which restoration model
  runs (both ship inside the `gfpgan` package already in `requirements-
  enhancer.txt`, so switching is a config change, not a new dependency, and
  both are Apache 2.0 / commercial-safe):
  - `gfpgan` (default) — GFPGANv1.4.
  - `restoreformer` — RestoreFormer; generally better identity preservation
    and detail than GFPGAN at similar speed, worth trying if GFPGAN's
    output looks too "smoothed"/beautified for your use case. Note
    `FACE_ENHANCER_WEIGHT` has no effect with this model (that parameter is
    specific to GFPGAN's architecture; RestoreFormer silently ignores it).

  A note on a commonly-recommended third option, **CodeFormer**: it's
  generally considered the strongest of the three, especially on
  occlusions like glasses, but its weights are licensed **non-commercial
  only** (S-Lab License 1.0 — commercial use requires contacting the
  authors) and it isn't a properly maintained pip package, so it isn't
  wired up here. Worth it if this deployment is genuinely non-commercial/
  internal and you're willing to vendor its architecture code; skip it
  otherwise.

**Glasses/eyewear look distorted or blurred after enhancement:** this is a
known GFPGAN limitation — its restoration model is trained mostly on bare
faces, and lens glare/frame edges commonly get misread as noise and
"corrected" away. Two settings address it, both on by default when the
enhancer is enabled:

- `PROTECT_EYEWEAR_REGION` (default `true`) — uses the face's eye
  landmarks to build a band over the glasses area and blends GFPGAN's
  output back toward the pre-enhancement swap result there, so the rest of
  the face still gets sharpened normally. See
  `app/core/face_engine.py:_eye_band_mask()` / `_apply_face_enhancer()`.
- `EYEWEAR_PROTECTION_STRENGTH` (default `0.6`, range `0-1`) — how strongly
  to protect that band. Raise it toward `1.0` if glasses are still visibly
  warped; lower it if the eye area now looks noticeably softer than the
  rest of the enhanced face.
- `FACE_ENHANCER_WEIGHT` (default `0.5`, range `0-1`) — GFPGAN's own
  restoration strength, independent of the eyewear band. Lowering it (e.g.
  `0.3`) makes GFPGAN's output closer to the raw swap everywhere, which
  can help if distortion isn't limited to the eyewear area.

If a swap still looks off after all of the above:
- **Wrong face picked** — tune `FACE_MATCH_THRESHOLD` (see above).
- **Blurry/low-detail result** — turn on `ENABLE_FACE_ENHANCER`, and use a
  sharp, well-lit, front-facing `SwapSource` photo; output quality is
  bottlenecked by the source photo's quality as much as by any setting here.
- **Visible seam/edge around the face** — this is what color correction
  targets; confirm `ENABLE_COLOR_CORRECTION=true` and it's not being
  swallowed by a stale `.env`.

## Performance / scaling notes

- **Model loading is lazy and cached per worker process**
  (`app/core/face_engine.py`) — the first job a given worker picks up pays
  the model-load cost, every job after that in the same process reuses the
  already-loaded model in memory.
- **Video is the bottleneck.** Each frame runs through detection + swap.
  For long videos, consider: processing on a frame-skip + interpolation
  schedule, capping max video length/resolution on upload, or simply
  running more `app/worker.py` processes (`docker compose up --scale
  worker=N`) — RabbitMQ round-robins jobs across however many workers are
  consuming the queue, so this is the main lever for throughput.
- The status cache (`app/store.py`) is Redis, shared between the API and
  every worker process/machine, with a TTL (`REQUEST_RECORD_TTL_SECONDS`)
  so old job records don't accumulate forever.
- `inswapper_128` is a **one-shot** model — no per-face-pair training step,
  which is what makes it suitable for an on-demand public-facing service
  (compare to DeepFaceLab, which needs hours of training per face pair and
  fits offline/VFX workflows better than live web traffic).

## Responsible use

Consider adding, depending on your jurisdiction and audience: visible
watermarking of outputs, content moderation on uploads, rate limiting, and
logging/audit trails. Non-consensual use of this kind of tool carries real
legal exposure in a growing number of jurisdictions (e.g. the U.S. TAKE IT
DOWN Act) — worth a compliance review before launch, not after.

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

**Model load is slow on every job, not just the first one**
That means something is re-creating the `FaceAnalysis`/swapper objects
instead of reusing the cached ones in `face_engine.py` — check you're not
restarting the worker process between jobs; each new worker process pays
the load cost once, on its first job.

**Jobs stay stuck on "Starting"**
Means nothing is consuming the queue — check `app/worker.py` is actually
running and connected to the same `RABBITMQ_URL` as the API. The RabbitMQ
management UI (`http://localhost:15672` with docker-compose) shows queue
depth and connected consumers, which is the fastest way to confirm.

**Worker connects then immediately gets disconnected (`IncompatibleProtocolError` / EOF during handshake)**
Usually a startup race, not a real failure: RabbitMQ's TCP port can start
accepting connections a moment before its AMQP listener is fully ready
(especially noticeable through Docker Desktop's Windows/WSL2 port
forwarding). Confirm with `docker compose ps` (should show `healthy`) and
`docker compose logs rabbitmq` (look for `Server startup complete`), then
just retry starting the worker.

**Job fails with "No female face was detected" or "Could not find the person from TargetSource"**
Either the default (no `TargetSource`) fallback couldn't find any female
face in `OriginalSource`, or the person in `TargetSource` genuinely isn't
in `OriginalSource` closely enough per `FACE_MATCH_THRESHOLD`. Try lowering
`FACE_MATCH_THRESHOLD` slightly (e.g. `0.3`) if you're confident the person
is in frame but the pose/lighting/angle differs a lot between
`TargetSource` and `OriginalSource`.

**Worker crashes mid-job with `PRECONDITION_FAILED - delivery acknowledgement ... timed out`**
RabbitMQ 3.8+ force-closes a channel if a delivered message isn't acked
within `consumer_timeout` (default 30 minutes) — this is a broker-side
safety net, separate from the heartbeat handling described in
`app/worker.py`'s threading note (the swap itself was still running fine in
the background thread; it just couldn't ack in time). A CPU video job with
`ENABLE_FACE_ENHANCER=true` can easily take longer than 30 minutes.
`rabbitmq.conf` (mounted into the `rabbitmq` service in both
`docker-compose.yml` and `docker-compose.prod.yml`) raises this to 6 hours —
recreate the `rabbitmq` container after pulling this change
(`docker compose up -d --force-recreate rabbitmq`) for it to take effect. If
you're deploying with `docker-compose.prod.yml`, remember to copy
`rabbitmq.conf` onto the server alongside it.

If jobs are hitting this at all, it's worth checking *why* a job is taking
that long in the first place — two common causes, both visible in the
worker's per-frame log lines:
- **Silent CPU fallback**: if you set `EXECUTION_PROVIDER=cuda` but the log
  shows `Applied providers: ['CPUExecutionProvider']` and/or "GPU requested
  but CUDA is unavailable", onnxruntime couldn't load its CUDA provider
  (commonly a missing/mismatched CUDA/cuDNN DLL — see "GPU not picked up"
  above) and silently ran the whole job on CPU instead, which is 10-20x
  slower and the main reason a job would run long enough to hit this
  timeout at all.
- **Per-frame time climbing over the course of the job** (e.g. 1s/frame at
  the start, 15-20s/frame by the middle): consistent with memory pressure
  on a CPU-only run holding `buffalo_l` + `inswapper_128` + GFPGAN in memory
  simultaneously. The frame loop now calls `gc.collect()` periodically as
  cheap insurance (see `GC_EVERY_N_FRAMES` in `app/core/video_swap.py`); if
  it's still climbing after that, it's genuine system memory pressure —
  check Task Manager, close other memory-heavy applications, or (biggest
  lever) turn off `ENABLE_FACE_ENHANCER` for video on CPU.

## Extending this

- **Temporal smoothing for video**: the current implementation swaps each
  frame independently, which can flicker slightly on shaky/low-quality
  footage. A face-tracking pass (carry the previous frame's bounding box
  forward instead of re-detecting from scratch every frame) is the next
  upgrade if you see this in practice.
- **Multi-person swaps in one request**: right now exactly one person per
  `OriginalSource` gets swapped (chosen via `TargetSource` or the female
  fallback). Swapping multiple different people in the same photo/video
  with different `SwapSource` faces would mean accepting a list of
  `(TargetSource, SwapSource)` pairs instead of a single pair.
- **Dead-lettering**: `app/worker.py` currently acks every message after
  processing, even on failure, and records the failure reason in Redis
  rather than requeuing. If you want RabbitMQ itself to distinguish
  "bad input, don't retry" from "infra hiccup, retry a few times," wire up
  a dead-letter exchange and have the worker `basic_nack` on infra-type
  errors specifically.
- **Push instead of poll**: if you'd rather not poll `GET /api/swap/{job_id}`,
  the natural next step is a webhook — add a `CallbackUrl` to `SwapRequest`
  and have `app/worker.py` POST status updates to it as they happen.
- **Caller-supplied idempotency key**: if you need retry-safe de-duplication
  again (e.g. a caller-supplied `TransId` that maps to a `job_id`), that's a
  small addition on top of the current job_id-only design — add the field
  back to `SwapRequest`, and look it up in Redis before generating a new
  `job_id` in `app/main.py`.
