from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    execution_provider: str = "cpu"           # "cuda" or "cpu"
    swapper_model_path: str = "models/inswapper_128.onnx"
    enable_face_enhancer: bool = False

    max_image_mb: int = 15
    max_video_mb: int = 300

    # Used for internal job bookkeeping only (not part of the shared-folder
    # contract with the website/mobile server).
    storage_dir: str = "storage"

    # The shared folder contract: the website/mobile server drops
    # OriginalSource + SwapSource files into uploads_dir, this service reads
    # them from there, and writes the processed result into outputs_dir.
    # Backed by *_raw fields (not the same name as the property) on purpose —
    # a pydantic field and a @property can't safely share one name. See
    # uploads_dir / outputs_dir properties below for the actual paths used
    # elsewhere in the app.
    uploads_dir_raw: str = Field(default="storage/uploads", alias="UPLOADS_DIR")
    outputs_dir_raw: str = Field(default="storage/outputs", alias="OUTPUTS_DIR")

    class Config:
        env_file = ".env"
        populate_by_name = True

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
        return self.execution_provider.lower() == "cuda"

    @property
    def onnx_providers(self) -> list:
        if self.use_cuda:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


settings = Settings()
