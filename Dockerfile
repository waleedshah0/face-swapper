# GPU-enabled Dockerfile for the Face Swap Studio app.
# This image is intended to run on a host with NVIDIA drivers and a supported GPU.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY requirements.txt requirements-enhancer.txt .env ./
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir --retries 10 --timeout 600 -r requirements.txt

ARG INSTALL_ENHANCER=false
RUN if [ "$INSTALL_ENHANCER" = "true" ] ; then \
    python3 -m pip install --no-cache-dir -r requirements-enhancer.txt ; \
    fi

COPY . .

# Ensure the ONNX swapper model is available in ./models at runtime.
# You can mount it with -v /host/path/models:/srv/app/models or copy it into the repo.

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
