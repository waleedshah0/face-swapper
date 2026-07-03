from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    execution_provider: str = Field(default="cpu", alias="EXECUTION_PROVIDER")
    swapper_model_path: str = Field(default="models/inswapper_128.onnx", alias="SWAPPER_MODEL_PATH")
    enable_face_enhancer: bool = Field(default=False, alias="ENABLE_FACE_ENHANCER")

    max_image_mb: int = Field(default=15, alias="MAX_IMAGE_MB")
    max_video_mb: int = Field(default=300, alias="MAX_VIDEO_MB")

    storage_dir: str = Field(default="storage", alias="STORAGE_DIR")
    uploads_dir_raw: str = Field(default="storage/uploads", alias="UPLOADS_DIR")
    outputs_dir_raw: str = Field(default="storage/outputs", alias="OUTPUTS_DIR")

    face_analyser_name: str = Field(default="buffalo_l", alias="FACE_ANALYSER_NAME")
    face_detector_size: int = Field(default=640, alias="FACE_DETECTOR_SIZE")

    # RabbitMQ: swap jobs are published here after payload validation and
    # picked up by app/worker.py, which does the actual (slow) swap work.
    rabbitmq_url: str = Field(default="amqp://guest:guest@localhost:5672/%2F", alias="RABBITMQ_URL")
    rabbitmq_swap_queue: str = Field(default="swap_jobs", alias="RABBITMQ_SWAP_QUEUE")
    rabbitmq_prefetch_count: int = Field(default=1, alias="RABBITMQ_PREFETCH_COUNT")

    # Redis: shared status cache. The API writes "Starting" when a job is
    # enqueued; the worker updates it to "In progress" / "Completed" /
    # "Failed" as it processes the job. The SSE stream endpoint reads it.
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    request_record_ttl_seconds: int = Field(default=86400, alias="REQUEST_RECORD_TTL_SECONDS")

    @field_validator("execution_provider")
    @classmethod
    def normalize_execution_provider(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator(
        "max_image_mb",
        "max_video_mb",
        "face_detector_size",
        "rabbitmq_prefetch_count",
        "request_record_ttl_seconds",
    )
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Value must be greater than 0")
        return value

    @property
    def uploads_dir(self) -> Path:
        p = Path(self.uploads_dir_raw)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def outputs_dir(self) -> Path:
        p = Path(self.outputs_dir_raw)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def jobs_dir(self) -> Path:
        p = Path(self.storage_dir) / "jobs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def use_cuda(self) -> bool:
        return self.execution_provider == "cuda"

    @property
    def onnx_providers(self) -> list[str]:
        if self.use_cuda:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


settings = Settings()
