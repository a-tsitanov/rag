"""Tests for the ingestion task (taskiq).

Uses ``taskiq.InMemoryBroker`` so the suite runs without RabbitMQ.
Real AMQP round-trip is covered by the end-to-end smoke step in the
migration plan.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from taskiq import InMemoryBroker


# ── fake backends ────────────────────────────────────────────────────


class FakePG:
    def __init__(self):
        self.docs: dict[str, dict] = {}

    async def __call__(self, **kwargs) -> None:
        self.docs[kwargs["doc_id"]] = kwargs


class FakeMilvusWriter:
    def __init__(self):
        self.rows: list[dict] = []

    async def __call__(self, rows: list[dict]) -> None:
        self.rows.extend(rows)


class FakeLightRAG:
    def __init__(self):
        self.texts: list[str] = []

    async def __call__(self, text: str) -> None:
        self.texts.append(text)


def _fake_embed(texts):
    import numpy as np

    return np.ones((len(texts), 1024), dtype="float32") / (1024 ** 0.5)


# ── fixture: in-memory broker + ``process_document`` task ────────────


@pytest_asyncio.fixture
async def inmem_task():
    """Wire an AsyncDocumentWorker (with fake backends) into an
    InMemoryBroker-registered ``process_document`` task.

    Yields ``(task, pg, milvus, lightrag)``.
    """
    from pathlib import Path

    from src.ingestion.worker import AsyncDocumentWorker

    pg = FakePG()
    milvus = FakeMilvusWriter()
    lightrag = FakeLightRAG()
    worker = AsyncDocumentWorker(
        embed_fn=_fake_embed,
        milvus_writer=milvus,
        lightrag_inserter=lightrag,
        pg_status_updater=pg,
    )

    broker = InMemoryBroker()

    @broker.task
    async def process_document(
        doc_id: str, path: str, department: str, priority: str,
    ) -> None:
        result = await worker.process_document(
            Path(path), doc_id=doc_id, department=department,
        )
        if result.status == "failed":
            raise RuntimeError(result.error or "processing failed")

    await broker.startup()
    try:
        yield process_document, pg, milvus, lightrag
    finally:
        await broker.shutdown()


# ── happy path ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_kick_completes_successfully(inmem_task, tmp_path):
    task, pg, milvus, lightrag = inmem_task

    sample = tmp_path / "hello.txt"
    sample.write_text(
        "The enterprise knowledge base ingests many document types. "
        "We chunk text semantically and embed with BGE-M3."
    )

    doc_id = "11111111-1111-1111-1111-111111111111"
    handle = await task.kiq(
        doc_id=doc_id, path=str(sample),
        department="demo", priority="normal",
    )
    result = await handle.wait_result(timeout=10)

    assert result.is_err is False, getattr(result, "error", None)
    assert pg.docs[doc_id]["status"] == "completed"
    assert pg.docs[doc_id]["error"] in ("", None)
    assert len(milvus.rows) >= 1
    assert lightrag.texts == [sample.read_text()]


# ── failure path ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_kick_failure_writes_error(inmem_task):
    task, pg, _milvus, _rag = inmem_task

    doc_id = "22222222-2222-2222-2222-222222222222"
    handle = await task.kiq(
        doc_id=doc_id, path="/tmp/no_such_file_abcdef.pdf",
        department="", priority="low",
    )
    result = await handle.wait_result(timeout=10)

    assert result.is_err is True
    assert pg.docs[doc_id]["status"] == "failed"
    assert pg.docs[doc_id]["error"]


# ── priority mapping ─────────────────────────────────────────────────


def test_priority_value_mapping():
    from src.ingestion.tasks import priority_value

    assert priority_value("low") == 0
    assert priority_value("normal") == 5
    assert priority_value("high") == 9
    assert priority_value("bogus") == 5  # fallback
