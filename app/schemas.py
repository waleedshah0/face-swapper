from typing import Optional

from pydantic import BaseModel


class VideoJobCreated(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str               # pending | processing | done | failed
    progress: float
    error: Optional[str] = None
    download_url: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
