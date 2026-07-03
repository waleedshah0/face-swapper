"""
Standalone RabbitMQ consumer.

app/main.py only validates a swap request and publishes it to the
RABBITMQ_SWAP_QUEUE queue (see app/broker.py). This process is what
actually does the work: it consumes one job at a time (RABBITMQ_PREFETCH_COUNT
controls how many are pulled ahead of being acked), runs the existing
image/video swap pipeline, and keeps the job's status record in Redis
(app/store.py) up to date the whole way, keyed by job_id (the id
app/main.py generated when the job was queued — there is no separate
caller-supplied id):

    Starting  -[worker picks the job up]->  In progress  -> Completed
                                                  │                \\-> Failed (+ message)
                                                  └─ for video jobs, the message
                                                     field is updated roughly every
                                                     5% of progress (see
                                                     _make_progress_callback), so
                                                     GET /api/swap/{job_id} shows
                                                     real movement if you poll it
                                                     while a job is running.

Run one or more of these alongside the API:

    python -m app.worker

Multiple worker processes (on one machine or several) can consume from the
same queue concurrently — RabbitMQ round-robins deliveries between them, so
this is how you scale swap throughput horizontally without changing any code
here.

Threading note: pika's BlockingConnection needs its own thread to stay
responsive enough to send/receive heartbeats. A video swap can take many
minutes on CPU, so running it directly inside the AMQP message callback
starves that thread and RabbitMQ (or the OS) eventually kills the connection
as dead — the swap still finishes, but the subsequent ack fails and the
worker crashes. To avoid that, the actual swap runs in a background thread;
the main thread stays free to keep servicing the connection, and the ack is
handed back to it via `connection.add_callback_threadsafe()` once the swap
is done, which is pika's documented way to talk to a BlockingConnection from
another thread.
"""
from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path

from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection

from app.broker import SwapJobMessage, open_worker_channel
from app.config import settings
from app.core.face_engine import NoFaceFoundError
from app.core.image_swap import swap_image
from app.core.video_swap import swap_video
from app.store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
    RequestStore,
    RequestStoreError,
    get_store,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faceswap.worker")


_PROGRESS_STEP_PCT = 5.0  # publish a status update at most every 5% of progress


def _make_progress_callback(store: RequestStore, job: SwapJobMessage):
    """
    Build a progress_cb for swap_video(): throttled to roughly every 5% so a
    long video doesn't flood Redis with a write per frame, while still
    giving anyone polling GET /api/swap/{job_id} visible movement instead
    of silence between "In progress" and "Completed".
    """
    last_reported_pct = -1.0

    def _on_progress(pct: float) -> None:
        nonlocal last_reported_pct
        if pct < 100.0 and (pct - last_reported_pct) < _PROGRESS_STEP_PCT:
            return
        last_reported_pct = pct
        try:
            store.update_status(job.job_id, STATUS_IN_PROGRESS, message=f"{pct:.0f}% complete")
        except RequestStoreError:
            logger.warning("Could not publish progress update for job_id=%s (%.0f%%)", job.job_id, pct)

    return _on_progress


def _run_swap(store: RequestStore, job: SwapJobMessage) -> Path:
    """Resolve the job's file paths and run the same swap pipeline the old sync endpoint used."""
    original_path = settings.uploads_dir / job.original_source
    face_path = settings.uploads_dir / job.swap_source

    if not original_path.is_file():
        raise FileNotFoundError(f"OriginalSource not found: {job.original_source}")
    if not face_path.is_file():
        raise FileNotFoundError(f"SwapSource not found: {job.swap_source}")

    if job.media_type == "video":
        output_path = settings.outputs_dir / f"{job.job_id}.mp4"
        swap_video(
            original_path, face_path, output_path,
            progress_cb=_make_progress_callback(store, job),
        )
    else:
        # Images are sub-second to a couple of seconds — not worth the extra
        # Redis writes a progress callback would add.
        output_path = settings.outputs_dir / f"{job.job_id}.png"
        swap_image(original_path, face_path, output_path)

    return output_path


def _mark_completed(store: RequestStore, job: SwapJobMessage, output_path: Path) -> None:
    try:
        store.update_status(job.job_id, STATUS_COMPLETED, output_file=output_path.name)
    except RequestStoreError:
        logger.exception(
            "Swap for job_id=%s succeeded but the status cache couldn't be updated", job.job_id
        )
    logger.info("Completed job_id=%s -> %s", job.job_id, output_path.name)


def _mark_failed(store: RequestStore, job: SwapJobMessage, message: str) -> None:
    try:
        store.update_status(job.job_id, STATUS_FAILED, message=message)
    except RequestStoreError:
        logger.exception(
            "Swap for job_id=%s failed and the status cache couldn't be updated either", job.job_id
        )
    logger.warning("Failed job_id=%s: %s", job.job_id, message)


def _process_job(store: RequestStore, job: SwapJobMessage) -> None:
    logger.info("Picked up job_id=%s (%s)", job.job_id, job.media_type)

    try:
        store.update_status(job.job_id, STATUS_IN_PROGRESS)
    except RequestStoreError:
        logger.exception(
            "Could not mark job_id=%s as %r; processing anyway", job.job_id, STATUS_IN_PROGRESS
        )

    try:
        output_path = _run_swap(store, job)
    except NoFaceFoundError as exc:
        _mark_failed(store, job, str(exc))
    except Exception as exc:  # noqa: BLE001 - last line of defense so one bad job can't kill the worker
        logger.exception("Unexpected error processing job_id=%s", job.job_id)
        _mark_failed(store, job, f"Swap failed: {exc}")
    else:
        _mark_completed(store, job, output_path)


def _ack_threadsafe(connection: BlockingConnection, channel: BlockingChannel, delivery_tag: int) -> None:
    """
    Schedule an ack back onto the connection's own thread. Called from the
    background worker thread once a job is done — pika's BlockingConnection
    is not safe to call directly from any thread other than the one that
    created it, so add_callback_threadsafe() is the sanctioned way to hand
    work back to it.
    """
    def _do_ack() -> None:
        if channel.is_open:
            channel.basic_ack(delivery_tag=delivery_tag)

    try:
        connection.add_callback_threadsafe(_do_ack)
    except Exception:
        # Connection already closed (e.g. worker shutting down) — nothing
        # more we can do; RabbitMQ will redeliver this message once it
        # notices the consumer is gone.
        logger.exception("Could not schedule ack for delivery_tag=%s; connection may be closed", delivery_tag)


def _handle_delivery(
    store: RequestStore,
    connection: BlockingConnection,
    channel: BlockingChannel,
    delivery_tag: int,
    body: bytes,
) -> None:
    """Runs on a background thread so the connection's own thread stays free to send heartbeats."""
    try:
        job = SwapJobMessage.from_json(body.decode("utf-8"))
    except Exception:
        logger.exception("Dropping unparseable job message: %r", body[:200])
        _ack_threadsafe(connection, channel, delivery_tag)
        return

    try:
        _process_job(store, job)
    finally:
        # Ack regardless of outcome: failures are terminal states recorded
        # in Redis (status=Failed + message), not something a blind
        # RabbitMQ redelivery would fix. Wiring a dead-letter queue for
        # jobs that fail for infrastructure reasons (vs. bad input) is a
        # reasonable next step if that distinction matters in production.
        _ack_threadsafe(connection, channel, delivery_tag)


def _make_on_message(store: RequestStore, connection: BlockingConnection):
    def on_message(channel, method, properties, body):  # noqa: ANN001 - pika callback signature
        # Hand off immediately to a background thread and return — this
        # callback must stay fast so the connection's thread (this one)
        # keeps cycling back to process heartbeats while the swap runs.
        threading.Thread(
            target=_handle_delivery,
            args=(store, connection, channel, method.delivery_tag, body),
            daemon=True,
        ).start()

    return on_message


def main() -> None:
    store = get_store()
    connection, channel = open_worker_channel()
    channel.basic_consume(
        queue=settings.rabbitmq_swap_queue,
        on_message_callback=_make_on_message(store, connection),
    )

    def _handle_shutdown(signum, frame):  # noqa: ANN001
        logger.info("Shutdown signal received, finishing current job and stopping...")
        channel.stop_consuming()

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    logger.info(
        "Worker started. Listening on queue %r (prefetch=%d). Press Ctrl+C to stop.",
        settings.rabbitmq_swap_queue, settings.rabbitmq_prefetch_count,
    )
    try:
        channel.start_consuming()
    finally:
        if connection.is_open:
            connection.close()
        logger.info("Worker stopped.")


if __name__ == "__main__":
    main()
