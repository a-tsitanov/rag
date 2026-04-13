"""Document queue built on Redis Streams.

Provides durable, exactly-once delivery with consumer groups:

* ``DocumentProducer.add_document`` → XADD to ``documents:pending``
* ``DocumentConsumer.consume``      → XREADGROUP as async generator
* Automatic ACK on success, NACK → dead-letter after ``max_retries``
* ``get_job_status`` reads the status hash ``job:{job_id}``

Status lifecycle::

    pending → processing → done
                         ↘ failed  (after max_retries exhausted → DLQ)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────

STREAM_PENDING = "documents:pending"
STREAM_FAILED = "documents:failed"
CONSUMER_GROUP = "ingestion-workers"
STATUS_PREFIX = "job:"

# ── data types ────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


@dataclass
class Job:
    job_id: str
    stream_id: str  # Redis stream message ID (e.g. "1234567890-0")
    path: str
    department: str
    priority: str
    attempt: int = 0


# ── producer ──────────────────────────────────────────────────────────


class DocumentProducer:
    def __init__(self, redis: aioredis.Redis):
        self._r = redis

    async def add_document(
        self,
        path: str,
        department: str = "",
        priority: str = "normal",
    ) -> str:
        job_id = str(uuid.uuid4())

        stream_id = await self._r.xadd(
            STREAM_PENDING,
            {
                "job_id": job_id,
                "path": path,
                "department": department,
                "priority": priority,
            },
        )

        await self._r.hset(
            f"{STATUS_PREFIX}{job_id}",
            mapping={
                "status": JobStatus.pending.value,
                "path": path,
                "department": department,
                "priority": priority,
                "stream_id": stream_id if isinstance(stream_id, str)
                    else stream_id.decode(),
                "attempt": "0",
            },
        )

        logger.info(
            "Enqueued job %s  path=%s  department=%s  priority=%s",
            job_id, path, department, priority,
        )
        return job_id


# ── consumer ──────────────────────────────────────────────────────────


class DocumentConsumer:
    """Read from the ``documents:pending`` stream via a consumer group.

    Each instance picks a unique ``consumer_name``.  Call
    :meth:`consume` to get an async iterator of :class:`Job` objects.
    After processing, the caller **must** call :meth:`ack` (success) or
    :meth:`fail` (error).
    """

    def __init__(
        self,
        redis: aioredis.Redis,
        consumer_name: str | None = None,
        max_retries: int = 3,
        block_ms: int = 5000,
    ):
        self._r = redis
        self._consumer = consumer_name or f"worker-{uuid.uuid4().hex[:8]}"
        self._max_retries = max_retries
        self._block_ms = block_ms

    async def ensure_group(self) -> None:
        """Create the consumer group (idempotent)."""
        try:
            await self._r.xgroup_create(
                STREAM_PENDING, CONSUMER_GROUP, id="0", mkstream=True,
            )
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def consume(self) -> AsyncIterator[Job]:
        """Yield jobs from the stream, one at a time.

        The caller must ``await consumer.ack(job)`` or
        ``await consumer.fail(job, error)`` for each yielded job.
        """
        await self.ensure_group()

        while True:
            entries = await self._r.xreadgroup(
                CONSUMER_GROUP,
                self._consumer,
                {STREAM_PENDING: ">"},
                count=1,
                block=self._block_ms,
            )

            if not entries:
                continue

            for _stream_name, messages in entries:
                for stream_id, fields in messages:
                    if isinstance(stream_id, bytes):
                        stream_id = stream_id.decode()
                    fields = {
                        (k.decode() if isinstance(k, bytes) else k):
                        (v.decode() if isinstance(v, bytes) else v)
                        for k, v in fields.items()
                    }

                    job_id = fields["job_id"]

                    # read current attempt count
                    raw = await self._r.hget(
                        f"{STATUS_PREFIX}{job_id}", "attempt",
                    )
                    attempt = int(raw) if raw else 0

                    await self._r.hset(
                        f"{STATUS_PREFIX}{job_id}",
                        mapping={"status": JobStatus.processing.value,
                                 "attempt": str(attempt + 1)},
                    )

                    yield Job(
                        job_id=job_id,
                        stream_id=stream_id,
                        path=fields["path"],
                        department=fields.get("department", ""),
                        priority=fields.get("priority", "normal"),
                        attempt=attempt + 1,
                    )

    # ── ack / fail ────────────────────────────────────────────────────

    async def ack(self, job: Job) -> None:
        """Mark the job as successfully processed."""
        await self._r.xack(STREAM_PENDING, CONSUMER_GROUP, job.stream_id)
        await self._r.hset(
            f"{STATUS_PREFIX}{job.job_id}",
            mapping={"status": JobStatus.done.value},
        )
        logger.info("ACK job %s", job.job_id)

    async def fail(self, job: Job, error: str = "") -> None:
        """NACK the job.  Move to DLQ after ``max_retries``."""
        await self._r.xack(STREAM_PENDING, CONSUMER_GROUP, job.stream_id)

        if job.attempt >= self._max_retries:
            # dead-letter
            await self._r.xadd(
                STREAM_FAILED,
                {
                    "job_id": job.job_id,
                    "path": job.path,
                    "department": job.department,
                    "error": error,
                    "attempts": str(job.attempt),
                },
            )
            await self._r.hset(
                f"{STATUS_PREFIX}{job.job_id}",
                mapping={"status": JobStatus.failed.value, "error": error},
            )
            logger.warning(
                "Job %s moved to DLQ after %d attempts: %s",
                job.job_id, job.attempt, error,
            )
        else:
            # re-enqueue for retry
            await self._r.xadd(
                STREAM_PENDING,
                {
                    "job_id": job.job_id,
                    "path": job.path,
                    "department": job.department,
                    "priority": job.priority,
                },
            )
            await self._r.hset(
                f"{STATUS_PREFIX}{job.job_id}",
                mapping={"status": JobStatus.pending.value},
            )
            logger.info(
                "Re-enqueued job %s (attempt %d/%d)",
                job.job_id, job.attempt, self._max_retries,
            )


# ── status query ──────────────────────────────────────────────────────


async def get_job_status(
    redis: aioredis.Redis,
    job_id: str,
) -> JobStatus | None:
    raw = await redis.hget(f"{STATUS_PREFIX}{job_id}", "status")
    if raw is None:
        return None
    val = raw.decode() if isinstance(raw, bytes) else raw
    return JobStatus(val)
