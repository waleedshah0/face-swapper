from typing import Optional

from pydantic import BaseModel


class SwapAccepted(BaseModel):
    """Returned immediately by POST /api/swap once the job has been queued in RabbitMQ."""

    job_id: str                     # server-generated; the id everything else is tracked under
    site_id: str                    # caller-supplied; identifies which site/tenant this job belongs to
    media_type: str                 # "image" or "video"
    status: str                     # "Starting"


class SwapStatus(BaseModel):
    """
    A snapshot of one job's state, as stored in Redis and returned by
    GET /api/swap/{job_id}.
    """

    job_id: str
    site_id: str
    original_source: str
    swap_source: str
    target_source: Optional[str] = None   # reference photo of the specific person to swap, if given
    media_type: str
    status: str                     # "Starting" | "In progress" | "Completed" | "Failed"
    message: Optional[str] = None
    output_file: Optional[str] = None
    created_at: str
    updated_at: str


class ErrorResponse(BaseModel):
    detail: str
