"""
Minimal in-memory job tracker for the video swap endpoint.

Video swapping takes anywhere from several seconds to a few minutes
depending on length/resolution, so unlike the image endpoint it runs as a
background task: the client gets a job_id immediately, then polls
GET /api/jobs/{job_id} until status is "done" (or "failed").

This in-memory store is fine for a single-process demo/small deployment.
For multi-worker or multi-server deployments, swap this out for Redis or a
database table — the interface (create/get/update) would stay the same.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class Job:
    id: str
    status: str = "pending"          # pending | processing | done | failed
    progress: float = 0.0
    error: Optional[str] = None
    result_path: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


_jobs: Dict[str, Job] = {}
_lock = threading.Lock()


def create_job() -> Job:
    job = Job(id=str(uuid.uuid4()))
    with _lock:
        _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **fields) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
