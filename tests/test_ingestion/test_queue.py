"""Tests for the ingestion task (taskiq).

Uses ``taskiq.InMemoryBroker`` so the suite runs without RabbitMQ.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio
from langchain_core.embeddings import Embeddings
from taskiq import InMemoryBroker

from src.ingestion.chunker import SemanticChunker


# ── fake backends ────────────────────────────────────────────────────


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            (np.ones(768, dtype="float32") / (768 ** 0.5)).tolist()
            for _ in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FakeVectorStore:
    def __init__(self):
        self.docs: list[dict] = []

    def add_documents(self, documents, *, ids=None, **kw):
        for i, doc in enumerate(documents):
            self.docs.append({
                "id": ids[i] if ids else str(i),
                "content": doc.page_content,
                **doc.metadata,
            })
        return ids or [str(i) for i in range(len(documents))]


class FakePG:
    def __init__(self):
        self.docs: dict[str, dict] = {}

    async def __call__(self, **kwargs) -> None:
        self.docs[kwargs["doc_id"]] = kwargs


class FakeLightRAG:
    def __init__(self):
        self.texts: list[str] = []

    async def __call__(self, text: str) -> None:
        self.texts.append(text)


# ── fixture ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def inmem_task():
    from pathlib import Path

    from src.ingestion.worker import AsyncDocumentWorker

    pg = FakePG()
    vs = FakeVectorStore()
    lightrag = FakeLightRAG()
    emb = FakeEmbeddings()
    chunker = SemanticChunker(embeddings=emb, max_tokens=512, overlap=50)

    worker = AsyncDocumentWorker(
        vectorstore=vs,
        lightrag_inserter=lightrag,
        pg_status_updater=pg,
        chunker=chunker,
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
        yield process_document, pg, vs, lightrag
    finally:
        await broker.shutdown()


# ── tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_kick_completes_successfully(inmem_task, tmp_path):
    task, pg, vs, lightrag = inmem_task

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
    assert len(vs.docs) >= 1
    assert lightrag.texts == [sample.read_text()]


@pytest.mark.asyncio
async def test_kick_failure_writes_error(inmem_task):
    task, pg, _vs, _rag = inmem_task

    doc_id = "22222222-2222-2222-2222-222222222222"
    handle = await task.kiq(
        doc_id=doc_id, path="/tmp/no_such_file_abcdef.pdf",
        department="", priority="low",
    )
    result = await handle.wait_result(timeout=10)

    assert result.is_err is True
    assert pg.docs[doc_id]["status"] == "failed"


def test_priority_value_mapping():
    from src.ingestion.tasks import priority_value

    assert priority_value("low") == 0
    assert priority_value("normal") == 5
    assert priority_value("high") == 9
    assert priority_value("bogus") == 5
