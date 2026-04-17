"""Tests for HybridSearcher.

Uses a fake vectorstore that returns pre-seeded documents so the tests
don't need a real Milvus instance.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from langchain_core.documents import Document as LCDocument

from src.models.search import SearchResponse, SourceCitation
from src.retrieval.hybrid_search import HybridSearcher


# ── fakes ─────────────────────────────────────────────────────────────


class _StubRAG:
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


def _reranker(query: str, candidates: list[str]) -> list[float]:
    q_low = query.lower()
    scores = []
    for i, c in enumerate(candidates):
        base = 1.0 if q_low in c.lower() else 0.3
        scores.append(base + (len(candidates) - i) * 0.01)
    return scores


class FakeVectorStore:
    """Minimal stub with `similarity_search_with_score` interface."""

    def __init__(self, docs: list[tuple[LCDocument, float]]):
        self._docs = docs

    def similarity_search_with_score(
        self, query: str, k: int = 4, expr: str | None = None, **kwargs,
    ) -> list[tuple[LCDocument, float]]:
        results = self._docs
        if expr:
            results = [
                (d, s) for d, s in results
                if _matches_expr(d.metadata, expr)
            ]
        return results[:k]


def _matches_expr(meta: dict, expr: str) -> bool:
    """Naive filter: 'department == "X"' → check meta["department"] == "X"."""
    if "==" not in expr:
        return True
    key, val = expr.split("==", 1)
    key = key.strip().strip('"')
    val = val.strip().strip('"')
    return meta.get(key) == val


# ── test data ─────────────────────────────────────────────────────────

_DOCS = [
    LCDocument(
        page_content="kubernetes is a container orchestration platform used in production",
        metadata={"doc_id": "doc-1", "pk": "doc-1_0", "department": "engineering", "doc_type": "txt"},
    ),
    LCDocument(
        page_content="kubernetes networking relies on CNI plugins for pod communication",
        metadata={"doc_id": "doc-1", "pk": "doc-1_1", "department": "engineering", "doc_type": "txt"},
    ),
    LCDocument(
        page_content="marketing quarterly results exceeded expectations by 15%",
        metadata={"doc_id": "doc-2", "pk": "doc-2_0", "department": "marketing", "doc_type": "txt"},
    ),
    LCDocument(
        page_content="kubernetes cluster autoscaling improves resource utilization",
        metadata={"doc_id": "doc-1", "pk": "doc-1_2", "department": "engineering", "doc_type": "txt"},
    ),
]

_SCORED_DOCS = [(d, 0.9 - i * 0.1) for i, d in enumerate(_DOCS)]


# ── fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def searcher():
    vs = FakeVectorStore(_SCORED_DOCS)
    rag = _StubRAG()
    return HybridSearcher(
        rag=rag, vectorstore=vs, reranker_fn=_reranker,
    )


# ── tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_returns_response_with_sources(searcher):
    resp = await searcher.search(query="kubernetes", top_k=3)

    assert isinstance(resp, SearchResponse)
    assert resp.query == "kubernetes"
    assert resp.answer == "This is the synthesized answer."
    assert resp.latency_ms > 0
    assert len(resp.sources) <= 3
    for src in resp.sources:
        assert isinstance(src, SourceCitation)
        assert src.content


@pytest.mark.asyncio
async def test_department_filter_applied(searcher):
    resp = await searcher.search(
        query="kubernetes", department="engineering", top_k=5,
    )
    for src in resp.sources:
        assert src.department == "engineering"


@pytest.mark.asyncio
async def test_department_filter_excludes_others(searcher):
    resp = await searcher.search(
        query="kubernetes", department="marketing", top_k=5,
    )
    for src in resp.sources:
        assert src.department == "marketing"


@pytest.mark.asyncio
async def test_rerank_sorts_by_relevance(searcher):
    resp = await searcher.search(query="kubernetes", top_k=5)
    if len(resp.sources) >= 2:
        assert resp.sources[0].score >= resp.sources[1].score


@pytest.mark.asyncio
async def test_chunk_position_parsed(searcher):
    resp = await searcher.search(query="kubernetes", top_k=5)
    positions = {s.position for s in resp.sources}
    assert positions == {0, 1, 2}


@pytest.mark.asyncio
async def test_no_matching_department(searcher):
    resp = await searcher.search(
        query="kubernetes", department="nonexistent", top_k=5,
    )
    assert resp.sources == []
    assert resp.answer  # RAG still answers
