# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=1200 \
    PIP_RETRIES=20 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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

# Install the large Blackwell-compatible PyTorch wheels separately.
# Do not use --no-cache-dir here.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_ENHANCER" = "true" ]; then \
        python3 -m pip install \
          --retries 20 \
          --timeout 1200 \
          --index-url https://download.pytorch.org/whl/cu128 \
          torch==2.7.1+cu128 \
          torchvision==0.22.1+cu128; \
    fi

# Install the remaining project packages.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_ENHANCER" = "true" ]; then \
        python3 -m pip install \
          --retries 20 \
          --timeout 1200 \
          -r requirements.txt \
          -r requirements-enhancer.txt; \
    else \
        python3 -m pip install \
          --retries 20 \
          --timeout 1200 \
          -r requirements.txt; \
    fi

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]