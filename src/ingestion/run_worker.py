"""Runnable ingestion worker daemon.

Wires Redis Streams consumer → ``AsyncDocumentWorker`` → all storage
backends.  Run with::

    python -m src.ingestion.run_worker

    # or with a specific consumer name (for log tracing / ownership)
    CONSUMER_NAME=worker-prod-1 python -m src.ingestion.run_worker

Lifecycle per job::

    XREADGROUP              │  documents:pending
        │                    │  consumer group: ingestion-workers
        ▼
    process_document()      │  parse → chunk → embed → lightrag
                             │  → milvus.upsert → pg: status="completed"
        │
        ├── completed  → XACK
        └── failed     → NACK → retry (3×) → dead-letter ``documents:failed``
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import uuid
from pathlib import Path

import numpy as np
import psycopg
import redis.asyncio as aioredis

from src.config import settings
from src.ingestion.queue import DocumentConsumer
from src.ingestion.worker import AsyncDocumentWorker
from src.retrieval.lightrag_setup import close_rag, get_rag, init_rag
from src.storage.milvus_client import AsyncMilvusClient

logger = logging.getLogger(__name__)


# ── PostgreSQL upsert — Documents table ──────────────────────────────

_UPSERT_DOC = """
    INSERT INTO documents (id, path, status, department, processed_at)
    VALUES (
        %(id)s::uuid, %(path)s, %(status)s, %(department)s,
        CASE WHEN %(status)s IN ('completed', 'failed') THEN now() END
    )
    ON CONFLICT (id) DO UPDATE SET
        path         = EXCLUDED.path,
        status       = EXCLUDED.status,
        department   = EXCLUDED.department,
        processed_at = EXCLUDED.processed_at
"""


async def _run() -> None:
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    consumer_name = os.environ.get(
        "CONSUMER_NAME", f"worker-{uuid.uuid4().hex[:8]}",
    )
    logger.info("Starting worker %s", consumer_name)

    # ── connect to every backend ──────────────────────────────────
    redis = aioredis.from_url(settings.redis_url, decode_responses=False)
    await redis.ping()
    logger.info("redis connected")

    milvus = AsyncMilvusClient(
        uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
    )
    await milvus.connect()
    logger.info("milvus connected")

    pg = await psycopg.AsyncConnection.connect(
        settings.postgres_dsn, autocommit=True,
    )
    logger.info("postgres connected")

    await init_rag(graph_storage="Neo4JStorage")
    rag = get_rag()
    logger.info("lightrag ready (graph=Neo4JStorage)")

    # ── Ollama async embed ────────────────────────────────────────
    import ollama

    ollama_client = ollama.AsyncClient(host=settings.ollama_host)

    async def embed_fn(texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            r = await ollama_client.embeddings(
                model=settings.embedding_model, prompt=t,
            )
            vecs.append(r["embedding"])
        return np.array(vecs, dtype=np.float32)

    # ── storage callbacks wired into AsyncDocumentWorker ──────────
    async def milvus_writer(rows: list[dict]) -> None:
        await asyncio.to_thread(
            milvus._client.upsert,
            collection_name=milvus._collection,
            data=rows,
        )

    async def lightrag_insert(text: str) -> None:
        await rag.ainsert(text)

    async def pg_status(**kwargs) -> None:
        await pg.execute(_UPSERT_DOC, kwargs)

    processor = AsyncDocumentWorker(
        embed_fn=embed_fn,
        milvus_writer=milvus_writer,
        lightrag_inserter=lightrag_insert,
        pg_status_updater=pg_status,
    )

    consumer = DocumentConsumer(
        redis, consumer_name=consumer_name, block_ms=1_000,
    )

    # ── signal handling ───────────────────────────────────────────
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows

    # ── consume loop ──────────────────────────────────────────────
    logger.info("Waiting for jobs on stream 'documents:pending'...")
    consume_task = asyncio.create_task(_consume_loop(consumer, processor, stop))

    await stop.wait()
    logger.info("Shutdown signal — draining")
    consume_task.cancel()
    try:
        await consume_task
    except asyncio.CancelledError:
        pass

    # ── teardown ─────────────────────────────────────────────────
    logger.info("Closing connections")
    await milvus.disconnect()
    await pg.close()
    await close_rag()
    await redis.aclose()
    logger.info("Worker %s stopped", consumer_name)


async def _consume_loop(
    consumer: DocumentConsumer,
    processor: AsyncDocumentWorker,
    stop: asyncio.Event,
) -> None:
    async for job in consumer.consume():
        if stop.is_set():
            break

        logger.info(
            "job %s  path=%s  department=%s  attempt=%d",
            job.job_id, job.path, job.department, job.attempt,
        )

        try:
            result = await processor.process_document(Path(job.path))
            if result.status == "completed":
                await consumer.ack(job)
                logger.info(
                    "job %s done  chunks=%d  total=%.2fs",
                    job.job_id, result.chunks, result.total_s,
                )
            else:
                await consumer.fail(job, error=result.error or "unknown")
                logger.warning("job %s failed: %s", job.job_id, result.error)
        except Exception as exc:
            logger.exception("job %s crashed", job.job_id)
            try:
                await consumer.fail(job, error=str(exc))
            except Exception:
                logger.exception("failed to NACK job %s", job.job_id)


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
