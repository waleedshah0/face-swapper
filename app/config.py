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

    # Each restoration model is toggled independently via its own .env flag
    # — both ship inside the already-installed `gfpgan` package (no extra
    # dependency either way) and are Apache 2.0 (commercial-safe). Whether
    # the enhancer runs at all, and which model(s), is entirely decided by
    # these two flags (there is no separate master ENABLE_FACE_ENHANCER
    # switch anymore):
    #
    #   ENABLE_GFPGAN=true,  ENABLE_RESTOREFORMER=true  -> both run, chained:
    #       GFPGAN restores first, then RestoreFormer refines its output.
    #       Strongest result, slowest (two model passes per frame).
    #   ENABLE_GFPGAN=true,  ENABLE_RESTOREFORMER=false -> GFPGAN only.
    #   ENABLE_GFPGAN=false, ENABLE_RESTOREFORMER=true  -> RestoreFormer only
    #       (generally better identity preservation/detail than GFPGAN alone).
    #   ENABLE_GFPGAN=false, ENABLE_RESTOREFORMER=false -> no enhancement at
    #       all — raw swap output, fastest.
    #
    # See app/core/face_engine.py:_load_face_enhancers().
    enable_gfpgan: bool = Field(default=False, alias="ENABLE_GFPGAN")
    enable_restoreformer: bool = Field(default=False, alias="ENABLE_RESTOREFORMER")

    # inswapper_128 doesn't correct for skin-tone/lighting mismatch between
    # the pasted face and the frame it lands in ("pasted on" look). This
    # shifts the pasted face's color statistics to match its surroundings —
    # cheap (numpy/cv2 only, no extra model) so it defaults on. See
    # app/core/face_engine.py:_color_correct_pasted_face().
    enable_color_correction: bool = Field(default=True, alias="ENABLE_COLOR_CORRECTION")

    # GFPGAN's own restoration strength (0=barely touches the face, closer
    # to the raw swap; 1=maximum restoration). Lower this if GFPGAN is
    # visibly distorting faces wearing glasses or other accessories.
    face_enhancer_weight: float = Field(default=0.5, alias="FACE_ENHANCER_WEIGHT")

    # GFPGAN's face-restoration model is trained mostly on bare faces and
    # commonly warps/blurs eyeglasses (misreads lens glare/frame edges as
    # noise to "fix"). When true, the eye/glasses band of the face is
    # blended back toward the pre-enhancement swap result rather than fully
    # replaced by GFPGAN's output. See app/core/face_engine.py:_eye_band_mask().
    protect_eyewear_region: bool = Field(default=True, alias="PROTECT_EYEWEAR_REGION")

    # How strongly to protect the eye band from GFPGAN's changes: 0=no
    # protection (identical to GFPGAN's raw output), 1=eye band left exactly
    # as the swap produced it (no enhancer effect there at all).
    eyewear_protection_strength: float = Field(default=0.6, alias="EYEWEAR_PROTECTION_STRENGTH")

    max_image_mb: int = Field(default=15, alias="MAX_IMAGE_MB")
    max_video_mb: int = Field(default=300, alias="MAX_VIDEO_MB")

    storage_dir: str = Field(default="storage", alias="STORAGE_DIR")
    uploads_dir_raw: str = Field(default="storage/uploads", alias="UPLOADS_DIR")
    outputs_dir_raw: str = Field(default="storage/outputs", alias="OUTPUTS_DIR")

    face_analyser_name: str = Field(default="buffalo_l", alias="FACE_ANALYSER_NAME")
    face_detector_size: int = Field(default=640, alias="FACE_DETECTOR_SIZE")

    # Cosine-similarity threshold for deciding a detected face matches the
    # TargetSource reference photo (same person) rather than a bystander.
    # See app/core/face_engine.py:select_target_face(). Raise it if the
    # wrong face gets picked; lower it if the right face is being missed.
    face_match_threshold: float = Field(default=0.35, alias="FACE_MATCH_THRESHOLD")

    # RabbitMQ: swap jobs are published here after payload validation and
    # picked up by app/worker.py, which does the actual (slow) swap work.
    rabbitmq_url: str = Field(default="amqp://guest:guest@localhost:5672/%2F", alias="RABBITMQ_URL")
    rabbitmq_swap_queue: str = Field(default="swap_jobs", alias="RABBITMQ_SWAP_QUEUE")
    rabbitmq_prefetch_count: int = Field(default=1, alias="RABBITMQ_PREFETCH_COUNT")

    # Redis: shared status cache. The API writes "Starting" when a job is
    # enqueued; the worker updates it to "In progress" / "Completed" /
    # "Failed" as it processes the job. GET /api/swap/{job_id} reads it.
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    request_record_ttl_seconds: int = Field(default=86400, alias="REQUEST_RECORD_TTL_SECONDS")

    # Fired by app/worker.py once a job's status has been written to Redis
    # as Completed (100% done) — POSTs that job's cache record (job_id,
    # site_id, sources, status, output_file, timestamps, ...) as JSON,
    # best-effort (a failed/slow webhook never fails the job itself).
    #
    # The URL is per-SiteId: {site_id} in this template is substituted with
    # the job's own SiteId. E.g. with the default template, a job with
    # SiteId "csw102w" gets its webhook posted to
    # https://csw102w.cs4m.com/face_swap/response/. See
    # app/worker.py:_completion_webhook_url_for().
    #
    # Only the template lives in .env, so pointing at a different domain is
    # always a config edit, never a code edit. Leave empty to disable the
    # webhook entirely (app/worker.py:_send_completion_webhook() no-ops).
    completion_webhook_url_template: str = Field(
        default="https://{site_id}.cs4m.com/face_swap/response/",
        alias="COMPLETION_WEBHOOK_URL_TEMPLATE",
    )
    completion_webhook_timeout_seconds: int = Field(default=10, alias="COMPLETION_WEBHOOK_TIMEOUT_SECONDS")

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
        "completion_webhook_timeout_seconds",
    )
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Value must be greater than 0")
        return value

    @field_validator("face_match_threshold")
    @classmethod
    def validate_face_match_threshold(cls, value: float) -> float:
        if not (0.0 < value < 1.0):
            raise ValueError("FACE_MATCH_THRESHOLD must be between 0 and 1 (exclusive)")
        return value

    @field_validator("face_enhancer_weight", "eyewear_protection_strength")
    @classmethod
    def validate_unit_interval(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("Value must be between 0 and 1 (inclusive)")
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
    def enable_face_enhancer(self) -> bool:
        """True if at least one restoration model (GFPGAN and/or RestoreFormer) is enabled."""
        return self.enable_gfpgan or self.enable_restoreformer

    @property
    def use_cuda(self) -> bool:
        return self.execution_provider == "cuda"

    @property
    def onnx_providers(self) -> list[str]:
        if self.use_cuda:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


settings = Settings()
