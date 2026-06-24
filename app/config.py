from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    execution_provider: str = "cpu"           # "cuda" or "cpu"
    swapper_model_path: str = "models/inswapper_128.onnx"
    enable_face_enhancer: bool = False

    max_image_mb: int = 15
    max_video_mb: int = 300

    storage_dir: str = "storage"

    class Config:
        env_file = ".env"

    @property
    def uploads_dir(self) -> Path:
        p = Path(self.storage_dir) / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def outputs_dir(self) -> Path:
        p = Path(self.storage_dir) / "outputs"
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
