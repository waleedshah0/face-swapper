# CPU-only image — this is the default for this project since it has no
# GPU dependency. If you later get an NVIDIA GPU, see Dockerfile.gpu and
# switch requirements.txt to onnxruntime-gpu.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Place inswapper_128.onnx into ./models before building, or mount it as a
# volume at runtime — see README.md for download instructions.

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
