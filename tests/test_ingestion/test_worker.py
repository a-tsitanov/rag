"""Tests for AsyncDocumentWorker + BatchProcessor.

All external backends are replaced with in-memory fakes so the test
runs without any live services.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from langchain_core.embeddings import Embeddings

from src.ingestion.chunker import SemanticChunker
from src.ingestion.worker import (
    AsyncDocumentWorker,
    BatchProcessor,
    ProcessResult,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── fake backends ─────────────────────────────────────────────────────


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) & 0xFFFF_FFFF)
            v = rng.randn(768).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-10
            vecs.append(v.tolist())
        return vecs

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FakeVectorStore:
    """Mimics langchain Milvus.add_documents interface."""

    def __init__(self):
        self.docs: list[dict] = []

    def add_documents(self, documents, *, ids=None, **kwargs):
        for i, doc in enumerate(documents):
            self.docs.append({
                "id": ids[i] if ids else str(i),
                "content": doc.page_content,
                **doc.metadata,
            })
        return ids or [str(i) for i in range(len(documents))]


class FakeLightRAG:
    def __init__(self):
        self.texts: list[str] = []
        self.custom_kg_payloads: list[dict] = []

    async def insert(self, text: str) -> None:
        self.texts.append(text)

    async def insert_custom_kg(self, payload: dict) -> None:
        self.custom_kg_payloads.append(payload)


class FakePG:
    def __init__(self):
        self.docs: dict[str, dict] = {}

    async def update(self, **kwargs) -> None:
        self.docs[kwargs["doc_id"]] = kwargs


# ── fixture ───────────────────────────────────────────────────────────


@pytest.fixture
def backends():
    vs = FakeVectorStore()
    lightrag = FakeLightRAG()
    pg = FakePG()
    return vs, lightrag, pg


@pytest.fixture
def worker(backends):
    vs, lightrag, pg = backends
    emb = FakeEmbeddings()
    chunker = SemanticChunker(embeddings=emb, max_tokens=512, overlap=50)
    return AsyncDocumentWorker(
        vectorstore=vs,
        lightrag_inserter=lightrag.insert,
        lightrag_custom_kg_inserter=lightrag.insert_custom_kg,
        pg_status_updater=pg.update,
        chunker=chunker,
    )


# ── single-document tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_pdf(worker, backends):
    vs, lightrag, pg = backends

    result = await worker.process_document(FIXTURES / "sample.pdf")

    assert isinstance(result, ProcessResult)
    assert result.status == "completed"
    assert result.chunks >= 1
    assert result.error == ""
    assert result.doc_id
    assert result.total_s > 0

    step_names = [t.name for t in result.timings]
    assert step_names == ["parse", "chunk", "vectorstore", "lightrag", "pg_status"]
    for t in result.timings:
        assert t.elapsed_s >= 0

    assert len(vs.docs) == result.chunks
    for row in vs.docs:
        assert row["doc_id"] == result.doc_id
        assert row["content"]

    assert len(lightrag.texts) == 1
    assert pg.docs[result.doc_id]["status"] == "completed"


@pytest.mark.asyncio
async def test_process_docx(worker, backends):
    vs, lightrag, pg = backends
    result = await worker.process_document(FIXTURES / "sample.docx")

    assert result.status == "completed"
    assert result.chunks >= 1
    assert len(vs.docs) == result.chunks


@pytest.mark.asyncio
async def test_process_txt(worker, backends):
    _, _, pg = backends
    result = await worker.process_document(FIXTURES / "sample.txt")

    assert result.status == "completed"
    assert pg.docs[result.doc_id]["status"] == "completed"


@pytest.mark.asyncio
async def test_nonexistent_file_fails_gracefully(worker, backends):
    _, _, pg = backends
    result = await worker.process_document(Path("/tmp/no_such_file_12345.pdf"))

    assert result.status == "failed"
    assert result.error
    assert result.doc_id in pg.docs
    assert pg.docs[result.doc_id]["status"] == "failed"


# ── batch tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identifier_extraction_injects_custom_kg(worker, backends):
    """Stage C: identifiers get pre-injected as canonical KG nodes,
    and the doc text passed to ainsert is augmented with the canonical
    block (so the LLM uses canonical forms when extracting relations)."""
    _, lightrag, _ = backends

    result = await worker.process_document(FIXTURES / "contract.txt")
    assert result.status == "completed"

    # custom_kg payload should have been pushed once with several
    # canonical entities (deduped by (type, canonical))
    assert len(lightrag.custom_kg_payloads) == 1
    payload = lightrag.custom_kg_payloads[0]
    assert payload["chunks"] == []
    assert payload["relationships"] == []

    canonicals = {(e["entity_type"], e["entity_name"]) for e in payload["entities"]}
    assert ("ContractNumber", "ДП-2024/178-К") in canonicals
    assert ("DocumentDate", "2024-03-15") in canonicals
    assert ("INN", "7707083893") in canonicals
    assert ("PhoneNumber", "+74952345678") in canonicals
    assert ("Email", "i.ivanov@sevtech.ru") in canonicals
    assert ("Amount", "4250000.00 RUB") in canonicals
    assert any(t == "PostalAddress" for t, _ in canonicals)

    # All entities carry the doc_id as source_id and the file path
    for ent in payload["entities"]:
        assert ent["source_id"] == result.doc_id
        assert ent["file_path"].endswith("contract.txt")
        assert ent["description"]  # non-empty

    # The text given to ainsert must include the canonical block —
    # this is what teaches the LLM to use canonical entity_name when
    # building relationships.
    assert len(lightrag.texts) == 1
    augmented = lightrag.texts[0]
    assert "Канонические идентификаторы" in augmented
    assert "+74952345678" in augmented
    assert "ДП-2024/178-К" in augmented


@pytest.mark.asyncio
async def test_no_identifiers_skips_custom_kg(worker, backends):
    """Doc without identifiers must NOT trigger ainsert_custom_kg and
    must NOT augment the text — pure ainsert(text) path preserved."""
    _, lightrag, _ = backends

    result = await worker.process_document(FIXTURES / "sample.txt")
    assert result.status == "completed"
    assert lightrag.custom_kg_payloads == []
    # text passed verbatim — no augment block
    assert len(lightrag.texts) == 1
    assert "Канонические идентификаторы" not in lightrag.texts[0]


@pytest.mark.asyncio
async def test_batch_5_documents_parallel(worker, backends):
    vs, lightrag, pg = backends
    batch = BatchProcessor(worker, concurrency=10)

    paths = [
        FIXTURES / "sample.pdf",
        FIXTURES / "sample.docx",
        FIXTURES / "sample.pptx",
        FIXTURES / "sample.txt",
        FIXTURES / "sample.md",
    ]
    results = await batch.process_batch(paths, concurrency=10)

    assert len(results) == 5
    assert all(r.status == "completed" for r in results)

    doc_ids = [r.doc_id for r in results]
    assert len(set(doc_ids)) == 5

    total_chunks = sum(r.chunks for r in results)
    assert len(vs.docs) == total_chunks
    assert len(lightrag.texts) == 5
    assert len(pg.docs) == 5


@pytest.mark.asyncio
async def test_batch_respects_concurrency(worker, backends):
    concurrency_log: list[int] = []
    active = 0
    lock = asyncio.Lock()

    original_process = worker.process_document

    async def _tracked_process(path: Path) -> ProcessResult:
        nonlocal active
        async with lock:
            active += 1
            concurrency_log.append(active)
        try:
            return await original_process(path)
        finally:
            async with lock:
                active -= 1

    worker.process_document = _tracked_process

    paths = [FIXTURES / "sample.txt"] * 20
    batch = BatchProcessor(worker, concurrency=3)
    results = await batch.process_batch(paths, concurrency=3)

    assert all(r.status == "completed" for r in results)
    assert max(concurrency_log) <= 3
