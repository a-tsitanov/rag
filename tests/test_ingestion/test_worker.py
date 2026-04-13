"""Tests for AsyncDocumentWorker + BatchProcessor.

All external backends (Milvus, LightRAG, PostgreSQL) are replaced with
in-memory fakes so the test runs without any live services.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

from src.ingestion.worker import (
    AsyncDocumentWorker,
    BatchProcessor,
    ProcessResult,
)

FIXTURES = Path(__file__).parent / "fixtures"

# ── fake backends ─────────────────────────────────────────────────────


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic unit vectors — no model download."""
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) & 0xFFFF_FFFF)
        v = rng.randn(1024).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-10
        vecs.append(v)
    return np.array(vecs, dtype=np.float32)


class FakeMilvus:
    """Collects every row written to it."""

    def __init__(self):
        self.rows: list[dict] = []

    async def write(self, data: list[dict]) -> None:
        self.rows.extend(data)


class FakeLightRAG:
    """Records every text inserted."""

    def __init__(self):
        self.texts: list[str] = []

    async def insert(self, text: str) -> None:
        self.texts.append(text)


class FakePG:
    """Stores status updates as ``{doc_id: {...}}``."""

    def __init__(self):
        self.docs: dict[str, dict] = {}

    async def update(self, **kwargs) -> None:
        self.docs[kwargs["doc_id"]] = kwargs


# ── fixture ───────────────────────────────────────────────────────────


@pytest.fixture
def backends():
    milvus = FakeMilvus()
    lightrag = FakeLightRAG()
    pg = FakePG()
    return milvus, lightrag, pg


@pytest.fixture
def worker(backends):
    milvus, lightrag, pg = backends
    return AsyncDocumentWorker(
        embed_fn=_fake_embed,
        milvus_writer=milvus.write,
        lightrag_inserter=lightrag.insert,
        pg_status_updater=pg.update,
        embedding_batch_size=32,
        embedding_dim=1024,
    )


# ── single-document tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_pdf(worker, backends):
    milvus, lightrag, pg = backends

    result = await worker.process_document(FIXTURES / "sample.pdf")

    assert isinstance(result, ProcessResult)
    assert result.status == "completed"
    assert result.chunks >= 1
    assert result.error == ""
    assert result.doc_id  # non-empty UUID
    assert result.total_s > 0

    # timing steps recorded
    step_names = [t.name for t in result.timings]
    assert step_names == ["parse", "chunk", "embed", "lightrag", "milvus", "pg_status"]
    for t in result.timings:
        assert t.elapsed_s >= 0

    # backends received data
    assert len(milvus.rows) == result.chunks
    for row in milvus.rows:
        assert row["doc_id"] == result.doc_id
        assert len(row["embedding"]) == 1024
        assert row["content"]

    assert len(lightrag.texts) == 1
    assert pg.docs[result.doc_id]["status"] == "completed"


@pytest.mark.asyncio
async def test_process_docx(worker, backends):
    milvus, lightrag, pg = backends
    result = await worker.process_document(FIXTURES / "sample.docx")

    assert result.status == "completed"
    assert result.chunks >= 1
    assert len(milvus.rows) == result.chunks


@pytest.mark.asyncio
async def test_process_txt(worker, backends):
    milvus, _, pg = backends
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
async def test_batch_5_documents_parallel(worker, backends):
    milvus, lightrag, pg = backends
    batch = BatchProcessor(worker, concurrency=10)

    paths = [
        FIXTURES / "sample.pdf",
        FIXTURES / "sample.docx",
        FIXTURES / "sample.pptx",
        FIXTURES / "sample.txt",
        FIXTURES / "sample.md",
    ]
    results = await batch.process_batch(paths, concurrency=10)

    # all five completed
    assert len(results) == 5
    assert all(r.status == "completed" for r in results)

    # unique doc_ids
    doc_ids = [r.doc_id for r in results]
    assert len(set(doc_ids)) == 5

    # every doc has chunks in milvus
    total_chunks = sum(r.chunks for r in results)
    assert len(milvus.rows) == total_chunks

    # every doc went through lightrag
    assert len(lightrag.texts) == 5

    # every doc updated in PG
    assert len(pg.docs) == 5
    assert all(d["status"] == "completed" for d in pg.docs.values())

    # ── print metrics ─────────────────────────────────────────────
    print("\n\n=== Batch Processing Metrics (5 documents) ===\n")
    print(f"{'File':<20} {'Status':<10} {'Chunks':>6} {'Total':>8}   Steps")
    print("-" * 80)
    for r in results:
        name = Path(r.path).name
        steps = "  ".join(f"{t.name}={t.elapsed_s:.3f}s" for t in r.timings)
        print(f"{name:<20} {r.status:<10} {r.chunks:>6} {r.total_s:>7.3f}s   {steps}")

    grand_total = sum(r.total_s for r in results)
    wall_clock = max(r.total_s for r in results)  # parallel ≈ slowest
    print(f"\n  Sum of totals : {grand_total:.3f}s")
    print(f"  Wall-clock est: {wall_clock:.3f}s  (parallel)")
    print(f"  Total chunks  : {total_chunks}")
    print(f"  Milvus rows   : {len(milvus.rows)}")


@pytest.mark.asyncio
async def test_batch_respects_concurrency(worker, backends):
    """Verify semaphore actually limits parallelism."""
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
