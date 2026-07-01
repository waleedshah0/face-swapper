from typing import Optional

from pydantic import BaseModel


class SwapResult(BaseModel):
    status: str                     # "success"
    trans_id: Optional[str] = None
    media_type: str                 # "image" or "video"
    output_file: str                # filename written into settings.outputs_dir


class ErrorResponse(BaseModel):
    detail: str
