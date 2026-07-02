from typing import Optional

from pydantic import BaseModel


class SwapResult(BaseModel):
    status: str                     # "success"
    trans_id: Optional[str] = None
    media_type: str                 # "image" or "video"
    output_file: str                # filename written into settings.outputs_dir


class SwapValidationResponse(BaseModel):
    status: str
    message: str
    trans_id: Optional[str] = None
    original_source: Optional[str] = None
    swap_source: Optional[str] = None
    uploads_dir: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
