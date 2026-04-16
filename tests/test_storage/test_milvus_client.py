import random
import uuid

import pytest
import pytest_asyncio

from src.config import settings
from src.storage.milvus_client import (
    AsyncMilvusClient,
    Document,
    SearchResult,
)

DEPARTMENTS = ["engineering", "marketing", "sales", "hr"]
VECTOR_DIM = settings.ollama.embedding_dim


def _random_unit_vector(dim: int = VECTOR_DIM) -> list[float]:
    """Random vector normalised to unit length (required for COSINE)."""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def _make_documents(n: int) -> list[Document]:
    return [
        Document(
            id=str(uuid.uuid4()),
            content=f"Content for document {i}",
            embedding=_random_unit_vector(),
            doc_id=f"doc_{i // 10}",
            department=DEPARTMENTS[i % len(DEPARTMENTS)],
            created_at=1_700_000_000 + i * 3600,
            doc_type="text",
        )
        for i in range(n)
    ]


@pytest_asyncio.fixture
async def milvus(tmp_path):
    db_file = str(tmp_path / "test.db")
    # milvus-lite doesn't support HNSW; AUTOINDEX uses a compatible type
    client = AsyncMilvusClient(uri=db_file, collection="test_kb", index_type="AUTOINDEX")
    await client.connect()
    yield client
    await client.disconnect()


# ── upsert + search ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_insert_100_and_find_nearest(milvus):
    docs = _make_documents(100)

    await milvus.upsert_batch(docs)

    # Search with the first doc's own vector — must come back as top-1
    results = await milvus.search(query_vector=docs[0].embedding, top_k=5)

    assert 1 <= len(results) <= 5
    assert isinstance(results[0], SearchResult)
    assert results[0].id == docs[0].id
    # Scores are sorted descending (COSINE → higher = closer)
    assert results[0].score >= results[-1].score


# ── department filter ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_filtered_by_department(milvus):
    docs = _make_documents(100)
    await milvus.upsert_batch(docs)

    results = await milvus.search(
        query_vector=docs[0].embedding,
        top_k=50,
        department="engineering",
    )

    assert len(results) > 0
    assert all(r.department == "engineering" for r in results)


# ── upsert overwrites ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_updates_existing_document(milvus):
    vec = _random_unit_vector()
    doc_id = str(uuid.uuid4())

    original = Document(
        id=doc_id,
        content="Original",
        embedding=vec,
        doc_id="doc_0",
        department="engineering",
        created_at=1_700_000_000,
        doc_type="text",
    )
    await milvus.upsert_batch([original])

    updated = Document(
        id=doc_id,
        content="Updated",
        embedding=vec,
        doc_id="doc_0",
        department="marketing",
        created_at=1_700_000_001,
        doc_type="pdf",
    )
    await milvus.upsert_batch([updated])

    results = await milvus.search(query_vector=vec, top_k=1)

    assert len(results) == 1
    assert results[0].id == doc_id
    assert results[0].content == "Updated"
    assert results[0].department == "marketing"
    assert results[0].doc_type == "pdf"
