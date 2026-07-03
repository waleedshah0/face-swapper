"""
RabbitMQ integration.

publish_swap_job() is called from app/main.py, after payload validation, to
hand a swap job off to app/worker.py for processing. Kept intentionally
simple: one short-lived connection per publish, since swap requests aren't
frequent enough for connection setup to be the bottleneck (the swap itself
takes seconds to minutes). If publish volume ever gets high enough for that
to matter, swap this for a pooled/long-lived connection — publish_swap_job()
is the only call site that would need to change.

open_worker_channel() is used by app/worker.py, which is a long-running
process and keeps a single connection open for its whole lifetime.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Tuple

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPError

from app.config import settings

logger = logging.getLogger("faceswap.broker")


class BrokerError(Exception):
    """Raised when a message could not be published to / consumed from RabbitMQ."""


@dataclass
class SwapJobMessage:
    job_id: str                     # server-generated in app/main.py; the sole job identifier
    original_source: str
    swap_source: str
    media_type: str                 # "image" or "video"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "SwapJobMessage":
        return cls(**json.loads(raw))


def _connection_params() -> pika.URLParameters:
    return pika.URLParameters(settings.rabbitmq_url)


def _declare_queue(channel: BlockingChannel) -> None:
    # Durable so queued jobs survive a RabbitMQ restart; messages are
    # published as persistent (see publish_swap_job) to match.
    channel.queue_declare(queue=settings.rabbitmq_swap_queue, durable=True)


def publish_swap_job(message: SwapJobMessage) -> None:
    """Publish a swap job onto the queue. Blocking — call via run_in_threadpool from async code."""
    try:
        with pika.BlockingConnection(_connection_params()) as connection:
            channel = connection.channel()
            _declare_queue(channel)
            channel.basic_publish(
                exchange="",
                routing_key=settings.rabbitmq_swap_queue,
                body=message.to_json(),
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent,
                    content_type="application/json",
                ),
            )
    except AMQPError as exc:
        raise BrokerError(
            f"Could not publish swap job job_id={message.job_id!r} to RabbitMQ"
        ) from exc

    logger.info(
        "Published swap job job_id=%s -> queue %r",
        message.job_id, settings.rabbitmq_swap_queue,
    )


def open_worker_channel() -> Tuple[pika.BlockingConnection, BlockingChannel]:
    """
    Open a long-lived connection/channel for app/worker.py.

    prefetch_count is set from settings so a worker only pulls the next job
    once it has acked the current one (appropriate for CPU/GPU-heavy swap
    work — we don't want one worker process hoarding a batch of jobs while
    other workers sit idle). Caller owns the connection and must close it.
    """
    connection = pika.BlockingConnection(_connection_params())
    channel = connection.channel()
    _declare_queue(channel)
    channel.basic_qos(prefetch_count=settings.rabbitmq_prefetch_count)
    return connection, channel
