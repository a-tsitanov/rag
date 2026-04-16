"""Taskiq tasks + RabbitMQ broker for document ingestion.

This module is imported by **both** the FastAPI process (as a kicker —
it only calls ``process_document.kiq(...)``) and the ``taskiq worker``
CLI (which actually executes the task body).

Lifecycle
---------
* Module load: construct a single ``AioPikaBroker`` with a priority queue
  (``max_priority=10``) and a 3-retry middleware.
* Worker process: wire the dishka worker container via
  ``dishka.integrations.taskiq.setup_dishka`` so tasks can resolve
  ``FromDishka[AsyncDocumentWorker]``.  Close the container on
  ``TaskiqEvents.WORKER_SHUTDOWN``.
* API process: just uses the broker as a kicker — startup/shutdown is
  driven by the FastAPI lifespan in ``src.api.main``.

Run the worker with::

    taskiq worker src.ingestion.tasks:broker --workers 1
"""

from __future__ import annotations

from pathlib import Path

import psycopg
from dishka.integrations.taskiq import FromDishka, inject, setup_dishka
from loguru import logger
from taskiq import TaskiqEvents
from taskiq.middlewares import SimpleRetryMiddleware
from taskiq_aio_pika import AioPikaBroker

from src.config import settings
from src.di import build_worker_container
from src.ingestion.worker import AsyncDocumentWorker
from src.utils.logging import setup_logging

# ── broker ────────────────────────────────────────────────────────────

broker = AioPikaBroker(
    settings.rabbitmq.url,
    qos=settings.taskiq.prefetch,
    max_priority=10,
).with_middlewares(
    SimpleRetryMiddleware(default_retry_count=settings.taskiq.max_retries),
)


# ── DI wiring (worker process only) ──────────────────────────────────
#
# NOTE: we cannot rely on ``broker.is_worker_process`` at module-import
# time — the taskiq CLI only sets that flag *after* loading this module,
# right before ``broker.startup()``.  Hooking WORKER_STARTUP is the
# idiomatic way: the event fires exclusively on the worker side
# (broker.startup() dispatches either CLIENT_STARTUP or WORKER_STARTUP
# depending on the flag), and by then the dishka middleware can still
# be registered before the broker begins consuming.

_container: "object | None" = None


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def _wire_worker(_state) -> None:
    global _container
    # Настраиваем loguru именно тут: на module-load flag is_worker_process
    # ещё False — logger уйдёт в stdout вдвоём с taskiq-builtins. В
    # WORKER_STARTUP CLI уже проинициализировала процесс.
    setup_logging()
    _container = build_worker_container()
    setup_dishka(container=_container, broker=broker)
    logger.info("dishka worker container wired into taskiq broker")


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def _close_container(_state) -> None:
    global _container
    if _container is not None:
        logger.info("closing dishka worker container")
        await _container.close()
        _container = None


# ── task ──────────────────────────────────────────────────────────────


_IDEMPOTENT_STATUS_SELECT = (
    "SELECT status FROM documents WHERE id = %s::uuid"
)


@broker.task(
    retry_on_error=True,
    # hard-лимит на выполнение — получатель обернёт в anyio.fail_after
    timeout=settings.taskiq.task_timeout_s,
)
@inject(patch_module=True)
async def process_document(
    doc_id: str,
    path: str,
    department: str,
    priority: str,
    worker: FromDishka[AsyncDocumentWorker],
    pg: FromDishka[psycopg.AsyncConnection],
) -> None:
    """Parse → chunk → embed → LightRAG → Milvus for one document.

    Идемпотентно: если в Postgres уже ``status='completed'`` — skip.
    Защищает от re-delivery, когда воркер упал после `Milvus.upsert`,
    но до `ACK` RabbitMQ.

    On success the worker writes ``documents.status='completed'`` to
    Postgres.  On failure, taskiq's ``SimpleRetryMiddleware`` retries up
    to ``TASKIQ_MAX_RETRIES`` times; after terminal failure the worker
    writes ``status='failed'`` with the error string and RabbitMQ's DLX
    picks up the message.
    """
    # Привязываем job_id к контексту — все логи из worker.process_document,
    # Milvus-клиента, LightRAG и т.д. получают это поле "бесплатно".
    with logger.contextualize(job_id=doc_id):
        # ── idempotency check ────────────────────────────────────
        try:
            cur = await pg.execute(_IDEMPOTENT_STATUS_SELECT, (doc_id,))
            row = await cur.fetchone()
            if row and row[0] == "completed":
                logger.info("skip  reason=already_completed")
                return
        except Exception as exc:
            # PG недоступен — не блокируем процессинг, просто логируем
            logger.warning("idempotency check failed  error={err}", err=exc)

        logger.info(
            "process_document start  path={path} dept={dept} priority={priority}",
            path=path, dept=department, priority=priority,
        )
        result = await worker.process_document(
            Path(path), doc_id=doc_id, department=department,
        )
        if result.status == "failed":
            # re-raise so the retry middleware + DLX kick in
            raise RuntimeError(result.error or "processing failed")


# ── priority mapping ──────────────────────────────────────────────────

_PRIORITY_MAP = {"low": 0, "normal": 5, "high": 9}


def priority_value(name: str) -> int:
    """Map ``"low" | "normal" | "high"`` to a RabbitMQ priority int."""
    return _PRIORITY_MAP.get(name, 5)
