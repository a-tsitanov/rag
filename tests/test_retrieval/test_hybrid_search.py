"""End-to-end tests for HybridSearcher.

Uses:
  * milvus-lite (in-process) — real vector search, real department filter
  * stub LightRAG-like object — returns a canned answer
  * deterministic embedding + reranker functions — no model downloads
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import pytest_asyncio

from src.models.search import SearchResponse, SourceCitation
from src.retrieval.hybrid_search import HybridSearcher
from src.storage.milvus_client import AsyncMilvusClient, Document

# ── deterministic helpers ─────────────────────────────────────────────


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-10)


from src.config import settings as _settings


def _topic_vector(topic: str, dim: int = _settings.ollama.embedding_dim) -> np.ndarray:
    """Same topic string → same vector (so similarity actually works)."""
    rng = np.random.RandomState(hash(topic) & 0xFFFF_FFFF)
    return _unit(rng.randn(dim).astype(np.float32))


def _embed_fn(texts: list[str]) -> np.ndarray:
    """Simple topic-based embedder: each text's *first word* picks the vector.

    This ensures the test query ("kubernetes") matches documents whose
    content starts with "kubernetes".
    """
    vecs = []
    for t in texts:
        first = (t.split() or [""])[0].lower()
        vecs.append(_topic_vector(first))
    return np.array(vecs, dtype=np.float32)


def _reranker(query: str, candidates: list[str]) -> list[float]:
    """Fake cross-encoder — higher score if the query string appears verbatim
    in the candidate.  Adds small noise to force a distinct ordering."""
    q_low = query.lower()
    scores = []
    for i, c in enumerate(candidates):
        base = 1.0 if q_low in c.lower() else 0.3
        scores.append(base + (len(candidates) - i) * 0.01)
    return scores


class _StubRAG:
    """Minimal LightRAG-compatible object for tests."""

    def __init__(self, answer: str = "This is the synthesized answer."):
        self._answer = answer
        self.calls: list[dict] = []

    async def aquery(self, query: str, param) -> str:
        self.calls.append({
            "query": query,
            "mode": getattr(param, "mode", None),
            "top_k": getattr(param, "top_k", None),
        })
        return self._answer


# ── fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def milvus(tmp_path):
    client = AsyncMilvusClient(
        uri=str(tmp_path / "search.db"),
        collection="search_test",
        index_type="AUTOINDEX",  # milvus-lite doesn't support HNSW
    )
    await client.connect()

    # seed: 6 docs across 3 departments, topics: kubernetes/payments/invoices
    docs = [
        Document(
            id=f"doc_a_{i}",
            content="kubernetes orchestration cluster" if i == 0 else
                    "payments api gateway" if i == 1 else
                    "invoices financial reports",
            embedding=_topic_vector(
                "kubernetes" if i == 0 else
                "payments" if i == 1 else
                "invoices"
            ).tolist(),
            doc_id="doc-a",
            department="engineering" if i < 2 else "finance",
            created_at=1_700_000_000 + i,
            doc_type="pdf",
        )
        for i in range(3)
    ]
    # second set
    docs += [
        Document(
            id=f"doc_b_{i}",
            content=("kubernetes deployments engineering" if i == 0 else
                     "payments billing finance" if i == 1 else
                     "invoices quarterly"),
            embedding=_topic_vector(
                "kubernetes" if i == 0 else
                "payments" if i == 1 else
                "invoices"
            ).tolist(),
            doc_id="doc-b",
            department="engineering" if i == 0 else "finance",
            created_at=1_700_000_100 + i,
            doc_type="md",
        )
        for i in range(3)
    ]

    await client.upsert_batch(docs)
    yield client
    await client.disconnect()


@pytest.fixture
def stub_rag():
    return _StubRAG(answer="Synthesised answer about the query topic.")


@pytest.fixture
def searcher(milvus, stub_rag):
    return HybridSearcher(
        rag=stub_rag,
        milvus=milvus,
        embed_fn=_embed_fn,
        reranker_fn=_reranker,
        candidate_multiplier=3,
    )


# ── happy path ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_returns_response_with_sources(searcher, stub_rag):
    resp: SearchResponse = await searcher.search(
        query="kubernetes",
        mode="hybrid",
        top_k=5,
        user_id="u-1",
    )

    assert isinstance(resp, SearchResponse)
    assert resp.query == "kubernetes"
    assert resp.mode == "hybrid"
    assert resp.answer == "Synthesised answer about the query topic."
    assert resp.latency_ms > 0

    # sources: top rerank result should contain "kubernetes"
    assert len(resp.sources) > 0
    top = resp.sources[0]
    assert isinstance(top, SourceCitation)
    assert "kubernetes" in top.content.lower()
    assert top.score >= resp.sources[-1].score  # sorted desc

    # RAG was called with correct params
    assert len(stub_rag.calls) == 1
    assert stub_rag.calls[0]["query"] == "kubernetes"
    assert stub_rag.calls[0]["mode"] == "hybrid"


# ── department filter ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_department_filter_applied(searcher):
    resp = await searcher.search(
        query="kubernetes",
        department="engineering",
        top_k=10,
    )

    assert len(resp.sources) > 0
    assert all(s.department == "engineering" for s in resp.sources)


@pytest.mark.asyncio
async def test_department_filter_excludes_others(searcher):
    resp = await searcher.search(
        query="invoices",
        department="finance",
        top_k=10,
    )

    assert len(resp.sources) > 0
    assert all(s.department == "finance" for s in resp.sources)


# ── rerank actually reorders ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_rerank_sorts_by_relevance(searcher):
    """Candidates that contain the exact query term should rank above
    those that only share a topic vector."""
    resp = await searcher.search(query="kubernetes", top_k=5)

    # every "kubernetes" hit must come before any non-"kubernetes" hit
    seen_non_kube = False
    for s in resp.sources:
        has_kube = "kubernetes" in s.content.lower()
        if not has_kube:
            seen_non_kube = True
        else:
            assert not seen_non_kube, "rerank order is wrong"


# ── position parsed from chunk_id ─────────────────────────────────────


@pytest.mark.asyncio
async def test_chunk_position_parsed(milvus, stub_rag):
    # insert docs with chunk_id format "{doc_id}_{N}"
    docs = [
        Document(
            id=f"my-doc_{i}",
            content="kubernetes content",
            embedding=_topic_vector("kubernetes").tolist(),
            doc_id="my-doc",
            department="ops",
            created_at=1_700_000_000,
            doc_type="pdf",
        )
        for i in range(3)
    ]
    await milvus.upsert_batch(docs)

    searcher = HybridSearcher(
        rag=stub_rag, milvus=milvus,
        embed_fn=_embed_fn, reranker_fn=_reranker,
    )
    resp = await searcher.search(query="kubernetes", department="ops", top_k=5)

    positions = {s.position for s in resp.sources}
    assert positions == {0, 1, 2}


# ── empty-result safety ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_matching_department(searcher):
    resp = await searcher.search(
        query="kubernetes",
        department="nonexistent-department",
        top_k=5,
    )

    assert resp.answer  # RAG still ran
    assert resp.sources == []
    assert resp.latency_ms > 0
