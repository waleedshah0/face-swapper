"""
Redis-backed cache for face-swap request/job records.

Shared between the API process (app/main.py), which creates a record with
status STARTING right after payload validation and before publishing to
RabbitMQ, and the worker process (app/worker.py), which updates the same
record to IN_PROGRESS (with periodic progress in `message` for video jobs),
then COMPLETED or FAILED as it processes the job.

Keyed by job_id — the id app/main.py generates for every POST /api/swap and
hands back to the caller — so GET /api/swap/{job_id} is an O(1) lookup.
Records expire after settings.request_record_ttl_seconds so the cache
doesn't grow unbounded.

SiteId (caller-supplied) must be unique among active records — a second
POST /api/swap with a SiteId that's still cached is rejected with
SiteIdConflictError (see RequestStore.create()) rather than creating a
second job under it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import redis

from app.config import settings

logger = logging.getLogger("faceswap.store")

# --------------------------------------------------------------------------- #
# Status values. These are the only four states a request can be in.
# --------------------------------------------------------------------------- #
STATUS_STARTING = "Starting"
STATUS_IN_PROGRESS = "In progress"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"

TERMINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED}

_KEY_PREFIX = "faceswap:request:"

# Secondary index enforcing SiteId uniqueness: value is the job_id that
# claimed it, same TTL as the job record itself, so a SiteId frees up again
# once its job record would have expired anyway. See RequestStore.create().
_SITE_KEY_PREFIX = "faceswap:site:"


class RequestStoreError(Exception):
    """Raised when the status cache can't be reached or written to."""


class SiteIdConflictError(Exception):
    """Raised when a SiteId already has an active (non-expired) record in the cache."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RequestRecord:
    job_id: str
    site_id: str
    original_source: str
    swap_source: str
    media_type: str
    target_source: Optional[str] = None
    status: str = STATUS_STARTING
    message: Optional[str] = None
    output_file: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "RequestRecord":
        return cls(**json.loads(raw))


class RequestStore:
    """Thin wrapper around a Redis connection for request/job tracking."""

    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def _key(self, job_id: str) -> str:
        return f"{_KEY_PREFIX}{job_id}"

    def _site_key(self, site_id: str) -> str:
        return f"{_SITE_KEY_PREFIX}{site_id}"

    def site_id_exists(self, site_id: str) -> bool:
        """
        Fast-path check: does this SiteId already have an active record?
        Cheap (single Redis EXISTS), meant to let a caller reject an obvious
        duplicate before doing other work (file validation, etc.) — the
        authoritative, race-safe check is the atomic claim inside create().
        """
        try:
            return bool(self._client.exists(self._site_key(site_id)))
        except redis.RedisError as exc:
            raise RequestStoreError(f"Could not check SiteId={site_id!r} for uniqueness") from exc

    def create(
        self,
        job_id: str,
        site_id: str,
        original_source: str,
        swap_source: str,
        media_type: str,
        target_source: Optional[str] = None,
    ) -> RequestRecord:
        """
        Create (or overwrite) a record with status STARTING.

        SiteId must be unique among active (non-expired) records. Enforced
        here atomically via Redis SETNX (set-if-not-exists) rather than a
        separate check-then-set, so two concurrent requests with the same
        SiteId can't both slip through a race — whichever call reaches
        Redis first wins the claim, the other raises SiteIdConflictError.
        The claim shares this record's TTL, so the SiteId becomes available
        again once the record would have expired anyway.
        """
        try:
            claimed = self._client.set(
                self._site_key(site_id), job_id, nx=True, ex=self._ttl_seconds
            )
        except redis.RedisError as exc:
            raise RequestStoreError(f"Could not claim SiteId={site_id!r}") from exc
        if not claimed:
            raise SiteIdConflictError(f"SiteId '{site_id}' already exists.")

        record = RequestRecord(
            job_id=job_id,
            site_id=site_id,
            original_source=original_source,
            swap_source=swap_source,
            media_type=media_type,
            target_source=target_source,
            status=STATUS_STARTING,
        )
        self._save(record)
        return record

    def get(self, job_id: str) -> Optional[RequestRecord]:
        try:
            raw = self._client.get(self._key(job_id))
        except redis.RedisError as exc:
            raise RequestStoreError(f"Could not read status for job_id={job_id!r}") from exc

        if raw is None:
            return None
        try:
            return RequestRecord.from_json(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("Corrupt request record for job_id=%s: %s", job_id, exc)
            return None

    def update_status(
        self,
        job_id: str,
        status: str,
        message: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> Optional[RequestRecord]:
        """Update status/message/output_file on an existing record. No-op if missing."""
        record = self.get(job_id)
        if record is None:
            logger.warning("Tried to update status for unknown job_id=%s", job_id)
            return None

        record.status = status
        record.updated_at = _now_iso()
        if message is not None:
            record.message = message
        if output_file is not None:
            record.output_file = output_file

        self._save(record)
        return record

    def _save(self, record: RequestRecord) -> None:
        try:
            self._client.set(self._key(record.job_id), record.to_json(), ex=self._ttl_seconds)
        except redis.RedisError as exc:
            raise RequestStoreError(f"Could not save status for job_id={record.job_id!r}") from exc


_store: Optional[RequestStore] = None


def get_store() -> RequestStore:
    """Lazily construct a process-wide RequestStore (one Redis connection per process)."""
    global _store
    if _store is None:
        _store = RequestStore(settings.redis_url, settings.request_record_ttl_seconds)
    return _store
