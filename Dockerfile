# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=1200 \
    PIP_RETRIES=20 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gcc \
    g++ \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY requirements.txt requirements-enhancer.txt ./

RUN python3 -m pip install --upgrade pip setuptools wheel

ARG INSTALL_ENHANCER=true

# Install the large Blackwell-compatible PyTorch wheels separately, plus
# numpy and Cython up front — all three are build-time requirements for
# packages installed further down (basicsr needs torch, insightface compiles
# a Cython extension) and must already be present in the environment before
# that --no-build-isolation step runs, since with isolation off nothing gets
# auto-fetched into a throwaway build env anymore (see the note below).
# Do not use --no-cache-dir here.
#
# torch's cu128 wheels are enormous (the cudnn wheel alone is ~730MB, cublas
# ~610MB, etc.) — a multi-minute download even on a fast connection, and long
# enough that a single dropped/corrupted TCP chunk (SSL "decryption failed or
# bad record mac", a transient network issue, not a code bug) can kill the
# whole `pip install` outright; pip's own --retries mostly covers connection
# setup, not a stream that's already mid-transfer. The `until ... done` loop
# below just reruns the same pip install up to 5 times on failure — thanks to
# the cache mount, wheels that already finished downloading on a prior
# attempt are reused from /root/.cache/pip, so a retry only has to re-fetch
# whichever wheel actually got interrupted, not the whole set.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_ENHANCER" = "true" ]; then \
        i=0; \
        until python3 -m pip install \
                --retries 20 \
                --timeout 1200 \
                --index-url https://download.pytorch.org/whl/cu128 \
                torch==2.7.1+cu128 \
                torchvision==0.22.1+cu128; \
        do \
            i=$((i + 1)); \
            if [ "$i" -ge 5 ]; then \
                echo "pip install (torch/torchvision) failed after 5 attempts" >&2; \
                exit 1; \
            fi; \
            echo "pip install (torch/torchvision) attempt $i failed, retrying in 10s..." >&2; \
            sleep 10; \
        done && \
        python3 -m pip install \
          --retries 20 \
          --timeout 1200 \
          "numpy>=1.26.0,<2.0.0" \
          "Cython<3.1"; \
    fi

# Install the remaining project packages.
#
# --no-build-isolation on the enhancer branch: basicsr's setup.py does
# `import torch` at build time to detect CUDA support. Under normal build
# isolation, pip spins up a throwaway venv for that and resolves torch
# there completely unpinned/unrelated to our --index-url/version pin above
# — which as of pip's latest resolver means it downloads the newest torch
# release (a different, much larger CUDA-13-bundled build) from the
# default PyPI index just to run setup.py, and that multi-hundred-MB
# download is what was failing with IncompleteRead/ProtocolError. Passing
# --no-build-isolation makes pip build basicsr against the already-installed
# environment instead (our correctly pinned torch==2.7.1+cu128 and numpy
# from the step above), so no second unpinned torch is ever fetched.
#
# Side effect of --no-build-isolation applying to the WHOLE command: it also
# turns off build isolation for insightface, which compiles a Cython
# extension (mesh_core_cython.pyx) and was previously getting Cython
# auto-installed into its own isolated build env for free. With isolation
# off, Cython has to already be in the main environment instead — hence the
# explicit install above, alongside numpy, before this step runs. Pinned
# <3.1 since insightface 0.7.3 predates Cython 3's stricter language-level
# defaults and its old .pyx sources are more likely to build cleanly against
# the Cython 0.29.x/3.0.x line it was actually developed against.
#
# Both requirements files are still installed together in ONE invocation
# (not split into two separate pip calls) — splitting would let
# requirements-enhancer.txt's unpinned transitive numpy silently upgrade
# past requirements.txt's <2.0.0 pin in a second, independent resolve,
# which is exactly the numpy 1.x/2.x ABI break called out in
# requirements-enhancer.txt's own top-of-file comment.
#
# Same retry loop as above, same reasoning: this step's downloads are much
# smaller (torch/torchvision are already satisfied from the cache), but it's
# cheap insurance against the same class of transient network failure.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_ENHANCER" = "true" ]; then \
        i=0; \
        until python3 -m pip install \
                --retries 20 \
                --timeout 1200 \
                --no-build-isolation \
                -r requirements.txt \
                -r requirements-enhancer.txt; \
        do \
            i=$((i + 1)); \
            if [ "$i" -ge 5 ]; then \
                echo "pip install (requirements) failed after 5 attempts" >&2; \
                exit 1; \
            fi; \
            echo "pip install (requirements) attempt $i failed, retrying in 10s..." >&2; \
            sleep 10; \
        done; \
    else \
        i=0; \
        until python3 -m pip install \
                --retries 20 \
                --timeout 1200 \
                -r requirements.txt; \
        do \
            i=$((i + 1)); \
            if [ "$i" -ge 5 ]; then \
                echo "pip install (requirements) failed after 5 attempts" >&2; \
                exit 1; \
            fi; \
            echo "pip install (requirements) attempt $i failed, retrying in 10s..." >&2; \
            sleep 10; \
        done; \
    fi

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
