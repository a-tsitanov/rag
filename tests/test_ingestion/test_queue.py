"""Integration tests for DocumentQueue (Redis Streams).

Requires a running Redis on localhost:6379.
Tests are skipped automatically if Redis is unreachable.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
import redis.asyncio as aioredis

from src.ingestion.queue import (
    CONSUMER_GROUP,
    STATUS_PREFIX,
    STREAM_FAILED,
    STREAM_PENDING,
    DocumentConsumer,
    DocumentProducer,
    Job,
    JobStatus,
    get_job_status,
)

REDIS_URL = "redis://localhost:6379/0"

# ── unique stream names per test run (avoid collision) ────────────────

_ORIG_PENDING = STREAM_PENDING
_ORIG_FAILED = STREAM_FAILED


def _patch_streams(suffix: str):
    """Point module constants at test-specific stream names."""
    import src.ingestion.queue as qmod

    qmod.STREAM_PENDING = f"test:{suffix}:pending"
    qmod.STREAM_FAILED = f"test:{suffix}:failed"
    return qmod.STREAM_PENDING, qmod.STREAM_FAILED


def _restore_streams():
    import src.ingestion.queue as qmod

    qmod.STREAM_PENDING = _ORIG_PENDING
    qmod.STREAM_FAILED = _ORIG_FAILED


# ── fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def redis_conn():
    """Yield a live Redis connection, skip if unavailable."""
    r = aioredis.from_url(REDIS_URL, decode_responses=False)
    try:
        await r.ping()
    except Exception as exc:
        pytest.skip(f"Redis unavailable: {exc}")
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def streams(redis_conn, request):
    """Create test-scoped stream names and clean up afterwards."""
    suffix = request.node.name
    pending, failed = _patch_streams(suffix)

    yield pending, failed

    # cleanup
    await redis_conn.delete(pending, failed)
    # delete consumer group
    try:
        await redis_conn.xgroup_destroy(pending, CONSUMER_GROUP)
    except Exception:
        pass
    # clean up status hashes created during the test
    cursor = 0
    while True:
        cursor, keys = await redis_conn.scan(
            cursor, match=f"{STATUS_PREFIX}*", count=200,
        )
        if keys:
            await redis_conn.delete(*keys)
        if cursor == 0:
            break

    _restore_streams()


# ── producer ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_document_returns_job_id(redis_conn, streams):
    producer = DocumentProducer(redis_conn)
    job_id = await producer.add_document(
        path="/data/report.pdf", department="engineering", priority="high",
    )

    assert isinstance(job_id, str)
    assert len(job_id) == 36  # UUID format

    # status hash was created
    status = await get_job_status(redis_conn, job_id)
    assert status == JobStatus.pending


@pytest.mark.asyncio
async def test_multiple_documents_enqueued(redis_conn, streams):
    pending_stream, _ = streams
    producer = DocumentProducer(redis_conn)

    ids = []
    for i in range(5):
        jid = await producer.add_document(
            path=f"/data/doc_{i}.pdf", department="sales",
        )
        ids.append(jid)

    assert len(set(ids)) == 5

    # stream length matches
    length = await redis_conn.xlen(pending_stream)
    assert length == 5


# ── consumer: happy path ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consumer_reads_and_acks(redis_conn, streams):
    producer = DocumentProducer(redis_conn)
    consumer = DocumentConsumer(redis_conn, consumer_name="test-w1")

    # enqueue
    job_id = await producer.add_document(path="/data/a.pdf", department="hr")

    # consume one message
    received: list[Job] = []

    async def _drain():
        async for job in consumer.consume():
            received.append(job)
            await consumer.ack(job)
            return  # exit after first

    await asyncio.wait_for(_drain(), timeout=5)

    assert len(received) == 1
    assert received[0].job_id == job_id
    assert received[0].path == "/data/a.pdf"
    assert received[0].department == "hr"
    assert received[0].attempt == 1

    # status is done
    status = await get_job_status(redis_conn, job_id)
    assert status == JobStatus.done


# ── consumer: retry + DLQ ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_retries_then_dlq(redis_conn, streams):
    _, failed_stream = streams
    producer = DocumentProducer(redis_conn)
    consumer = DocumentConsumer(
        redis_conn, consumer_name="test-w2", max_retries=3, block_ms=1000,
    )

    job_id = await producer.add_document(path="/data/bad.pdf")

    attempts = 0

    async def _consume_all():
        nonlocal attempts
        async for job in consumer.consume():
            attempts += 1
            await consumer.fail(job, error=f"error #{attempts}")
            if attempts >= 3:
                return

    await asyncio.wait_for(_consume_all(), timeout=15)

    assert attempts == 3

    # status is failed
    status = await get_job_status(redis_conn, job_id)
    assert status == JobStatus.failed

    # DLQ has the message
    dlq_msgs = await redis_conn.xrange(failed_stream)
    assert len(dlq_msgs) >= 1

    last = dlq_msgs[-1][1]
    # decode if bytes
    last = {
        (k.decode() if isinstance(k, bytes) else k):
        (v.decode() if isinstance(v, bytes) else v)
        for k, v in last.items()
    }
    assert last["job_id"] == job_id
    assert last["attempts"] == "3"
    assert "error" in last


# ── three concurrent consumers ────────────────────────────────────────


@pytest.mark.asyncio
async def test_three_consumers_share_workload(redis_conn, streams):
    producer = DocumentProducer(redis_conn)

    n_jobs = 9
    for i in range(n_jobs):
        await producer.add_document(path=f"/data/doc_{i}.txt", department="ops")

    consumed_by: dict[str, list[str]] = {}

    async def _worker(name: str, count: int):
        consumer = DocumentConsumer(
            redis_conn, consumer_name=name, block_ms=1000,
        )
        done = 0
        async for job in consumer.consume():
            consumed_by.setdefault(name, []).append(job.job_id)
            await consumer.ack(job)
            done += 1
            if done >= count:
                return

    # 3 consumers, each tries to grab 3
    await asyncio.wait_for(
        asyncio.gather(
            _worker("w-0", 3),
            _worker("w-1", 3),
            _worker("w-2", 3),
        ),
        timeout=10,
    )

    all_ids = [jid for ids in consumed_by.values() for jid in ids]
    assert len(all_ids) == n_jobs
    assert len(set(all_ids)) == n_jobs  # no duplicates

    # each consumer got some work (not all to one)
    assert len(consumed_by) >= 2, (
        f"Expected ≥2 consumers to receive work, got: {list(consumed_by.keys())}"
    )

    # all jobs are done
    for jid in all_ids:
        assert await get_job_status(redis_conn, jid) == JobStatus.done


# ── get_job_status for unknown job ────────────────────────────────────


@pytest.mark.asyncio
async def test_get_status_unknown_job(redis_conn, streams):
    status = await get_job_status(redis_conn, "nonexistent-id")
    assert status is None
