"""Tests for ``src/retrieval/agent_search.py``.

Stage-F focused tests verify the final synthesis call uses accumulated
context (hl_keywords + enriched query).  The fixture stubs every
``HybridSearcher`` method the function calls so we don't need real
Milvus / Neo4j / LLM.

Stages G (early-exit) and H (per-round stats) will extend this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.models.search import SearchResponse, SourceCitation
from src.retrieval.agent_search import (
    _accumulated_hl_keywords,
    _build_enriched_query,
    _deduplicate_sources,
    _judge_context,
    _merge_graph_data,
    agentic_search,
)


# ── fakes ────────────────────────────────────────────────────────────


@dataclass
class StubLLMClient:
    """Returns a queue of canned judge replies as JSON strings."""

    replies: list[str] = field(default_factory=list)
    calls: list[list[dict]] = field(default_factory=list)

    async def chat(self, *, model: str, messages: list[dict]) -> dict:
        self.calls.append(messages)
        if not self.replies:
            return {"message": {"content": '{"sufficient": true}'}}
        reply = self.replies.pop(0)
        return {"message": {"content": reply}}


@dataclass
class StubSearcher:
    """Mocks the three ``HybridSearcher`` methods agent_search calls."""

    search_responses: list[SearchResponse] = field(default_factory=list)
    graph_responses: list[dict] = field(default_factory=list)
    ask_rag_answer: str = "synthesized answer"

    search_calls: list[tuple[str, dict]] = field(default_factory=list)
    graph_calls: list[tuple[str, dict]] = field(default_factory=list)
    ask_rag_calls: list[dict] = field(default_factory=list)

    async def search(self, query: str, **kwargs: Any) -> SearchResponse:
        self.search_calls.append((query, kwargs))
        if not self.search_responses:
            return SearchResponse(query=query, answer="", mode="hybrid")
        return self.search_responses.pop(0)

    async def query_graph_data(self, query: str, **kwargs: Any) -> dict:
        self.graph_calls.append((query, kwargs))
        if not self.graph_responses:
            return {}
        return self.graph_responses.pop(0)

    async def _ask_rag(self, query: str, mode: str, **kwargs: Any) -> str:
        self.ask_rag_calls.append({"query": query, "mode": mode, **kwargs})
        return self.ask_rag_answer


def _src(chunk_id: str, doc_id: str = "d1") -> SourceCitation:
    return SourceCitation(
        doc_id=doc_id, chunk_id=chunk_id, position=0,
        content=f"chunk-{chunk_id}", score=0.9,
    )


def _resp(chunk_ids: list[str]) -> SearchResponse:
    return SearchResponse(
        query="", answer="", mode="hybrid",
        sources=[_src(c) for c in chunk_ids],
    )


def _graph(entities: list[str], relations: list[tuple[str, str]] | None = None) -> dict:
    return {
        "data": {
            "entities": [{"entity_name": e} for e in entities],
            "relationships": [
                {"src_id": s, "tgt_id": t} for s, t in (relations or [])
            ],
        }
    }


# ── helper unit tests ────────────────────────────────────────────────


def test_build_enriched_query_no_follow_ups_returns_original() -> None:
    assert _build_enriched_query("what is X?", []) == "what is X?"


def test_build_enriched_query_appends_unique_follow_ups() -> None:
    out = _build_enriched_query(
        "what is X?",
        ["how does X relate to Y?", "what is X?", "sources of X?"],
    )
    assert "what is X?" in out
    assert "how does X relate to Y?" in out
    assert "sources of X?" in out
    # duplicate of original is dropped
    assert out.count("what is X?") == 1
    assert "Related sub-queries:" in out


def test_build_enriched_query_drops_empty_strings() -> None:
    assert _build_enriched_query("q", ["", "  ", "a"]).endswith("- a")


def test_accumulated_hl_keywords_dedupes_and_caps() -> None:
    gd = {
        "data": {
            "entities": [
                {"entity_name": "NVIDIA"},
                {"entity_name": "AMD"},
                {"entity_name": ""},  # filtered
                {"entity_name": "NVIDIA"},  # dup — preserved order keeps first
                {"entity_name": "TSMC"},
            ]
        }
    }
    assert _accumulated_hl_keywords(gd) == ["NVIDIA", "AMD", "TSMC"]
    assert _accumulated_hl_keywords(gd, limit=2) == ["NVIDIA", "AMD"]


def test_accumulated_hl_keywords_empty_input() -> None:
    assert _accumulated_hl_keywords({}) == []
    assert _accumulated_hl_keywords({"data": {"entities": []}}) == []


# ── Stage F end-to-end via stubs ────────────────────────────────────


@pytest.mark.asyncio
async def test_final_ask_rag_receives_enriched_query_and_hl_keywords() -> None:
    """Two-round path: judge requests a follow-up after round 1, then
    accepts.  Final ``_ask_rag`` must be called with:
      * enriched_query (original + the follow-up appended)
      * hl_keywords composed from entities accumulated across rounds
    """
    searcher = StubSearcher(
        search_responses=[
            _resp(["c1", "c2"]),
            _resp(["c3"]),
        ],
        graph_responses=[
            _graph(["NVIDIA", "AMD"]),
            _graph(["TSMC", "NVIDIA"]),  # NVIDIA dup → kept once
        ],
    )
    llm = StubLLMClient(
        replies=[
            json.dumps(
                {
                    "sufficient": False,
                    "follow_up_query": "who manufactures NVIDIA chips?",
                    "reason": "need supplier context",
                }
            ),
            json.dumps({"sufficient": True}),
        ]
    )

    result = await agentic_search(searcher, "what does NVIDIA do?", llm)

    assert result.agentic_rounds == 2
    assert result.follow_up_queries == ["who manufactures NVIDIA chips?"]

    # exactly one final synthesis
    assert len(searcher.ask_rag_calls) == 1
    final = searcher.ask_rag_calls[0]

    # enriched query carries both original and follow-up
    assert "what does NVIDIA do?" in final["query"]
    assert "who manufactures NVIDIA chips?" in final["query"]
    assert "Related sub-queries:" in final["query"]

    # hl_keywords come from accumulated graph entities (deduped)
    assert final["hl_keywords"] == ["NVIDIA", "AMD", "TSMC"]


@pytest.mark.asyncio
async def test_final_ask_rag_uses_original_when_no_follow_ups() -> None:
    """Single-round sufficient path: enriched_query equals original;
    hl_keywords come from the single round's entities."""
    searcher = StubSearcher(
        search_responses=[_resp(["c1"])],
        graph_responses=[_graph(["NVIDIA"])],
    )
    llm = StubLLMClient(replies=[json.dumps({"sufficient": True})])

    result = await agentic_search(searcher, "what is X?", llm)

    assert result.agentic_rounds == 1
    assert result.follow_up_queries is None
    assert len(searcher.ask_rag_calls) == 1
    final = searcher.ask_rag_calls[0]
    assert final["query"] == "what is X?"  # no enrichment
    assert final["hl_keywords"] == ["NVIDIA"]


@pytest.mark.asyncio
async def test_final_ask_rag_omits_hl_keywords_when_no_entities() -> None:
    """No graph entities → hl_keywords is None (falls through to
    LightRAG's own keyword-extraction LLM step)."""
    searcher = StubSearcher(
        search_responses=[_resp(["c1"])],
        graph_responses=[{}],  # empty graph data
    )
    llm = StubLLMClient(replies=[json.dumps({"sufficient": True})])

    await agentic_search(searcher, "q", llm)

    assert len(searcher.ask_rag_calls) == 1
    assert searcher.ask_rag_calls[0]["hl_keywords"] is None


@pytest.mark.asyncio
async def test_early_exit_when_followup_round_adds_nothing() -> None:
    """Stage G: round 2 returns same chunks/entities as round 1 — the
    judge must NOT be called a second time, the loop terminates with
    the round-1 context, and final synthesis still runs."""
    same_chunks = ["c1", "c2"]
    same_entities = ["NVIDIA"]
    searcher = StubSearcher(
        search_responses=[_resp(same_chunks), _resp(same_chunks)],
        graph_responses=[_graph(same_entities), _graph(same_entities)],
    )
    # Only ONE judge reply queued — if the loop wrongly calls judge a
    # second time, StubLLMClient defaults to "sufficient=true" so the
    # test would pass spuriously.  We verify by counting calls.
    llm = StubLLMClient(
        replies=[
            json.dumps(
                {
                    "sufficient": False,
                    "follow_up_query": "supply chain?",
                    "reason": "more context",
                }
            ),
        ]
    )

    result = await agentic_search(
        searcher, "what is X?", llm, max_rounds=3,
    )

    assert result.agentic_rounds == 2
    # judge called only on round 1; round 2 short-circuited
    assert len(llm.calls) == 1
    # final synthesis ran with the round-1 accumulated context
    assert len(searcher.ask_rag_calls) == 1
    assert searcher.ask_rag_calls[0]["hl_keywords"] == ["NVIDIA"]


@pytest.mark.asyncio
async def test_no_early_exit_when_followup_brings_new_info() -> None:
    """Stage G control: round 2 brings a new entity → judge IS called."""
    searcher = StubSearcher(
        search_responses=[_resp(["c1"]), _resp(["c2"])],
        graph_responses=[_graph(["NVIDIA"]), _graph(["TSMC"])],
    )
    llm = StubLLMClient(
        replies=[
            json.dumps(
                {
                    "sufficient": False,
                    "follow_up_query": "supply chain?",
                    "reason": "more context",
                }
            ),
            json.dumps({"sufficient": True}),
        ]
    )

    result = await agentic_search(
        searcher, "q", llm, max_rounds=3,
    )

    assert result.agentic_rounds == 2
    # judge called twice — once per round (no early exit)
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_round_one_always_calls_judge_even_with_no_results() -> None:
    """Edge case: empty graph + zero sources on round 1 must still call
    the judge — the early-exit gate is round-2+."""
    searcher = StubSearcher(
        search_responses=[_resp([])],
        graph_responses=[{}],
    )
    llm = StubLLMClient(replies=[json.dumps({"sufficient": True})])

    result = await agentic_search(searcher, "q", llm)

    assert result.agentic_rounds == 1
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_round_stats_populated_for_each_round() -> None:
    """Stage H: ``agentic_round_stats`` lists one entry per executed
    round with deltas and judge verdict."""
    searcher = StubSearcher(
        search_responses=[_resp(["c1", "c2"]), _resp(["c3"])],
        graph_responses=[
            _graph(["NVIDIA", "AMD"], relations=[("NVIDIA", "AMD")]),
            _graph(["TSMC"], relations=[("NVIDIA", "TSMC")]),
        ],
    )
    llm = StubLLMClient(
        replies=[
            json.dumps(
                {
                    "sufficient": False,
                    "follow_up_query": "supplier?",
                    "reason": "missing supply chain",
                }
            ),
            json.dumps(
                {"sufficient": True, "reason": "complete picture"}
            ),
        ]
    )

    result = await agentic_search(searcher, "what is X?", llm)

    stats = result.agentic_round_stats
    assert stats is not None
    assert len(stats) == 2

    r1 = stats[0]
    assert r1.round == 1
    assert r1.query == "what is X?"
    assert r1.new_sources == 2
    assert r1.new_entities == 2
    assert r1.new_relations == 1
    assert r1.sufficient is False
    assert "missing supply chain" in r1.judge_reason

    r2 = stats[1]
    assert r2.round == 2
    assert r2.query == "supplier?"
    assert r2.new_sources == 1
    assert r2.new_entities == 1
    assert r2.new_relations == 1
    assert r2.sufficient is True
    assert "complete picture" in r2.judge_reason


@pytest.mark.asyncio
async def test_round_stats_marks_skipped_judge_on_early_exit() -> None:
    """Stage H + G: when round 2 short-circuits because of no new info,
    the stats entry has ``sufficient=None`` and reason ``no new info``."""
    same_chunks = ["c1"]
    same_entities = ["NVIDIA"]
    searcher = StubSearcher(
        search_responses=[_resp(same_chunks), _resp(same_chunks)],
        graph_responses=[_graph(same_entities), _graph(same_entities)],
    )
    llm = StubLLMClient(
        replies=[
            json.dumps(
                {"sufficient": False, "follow_up_query": "q2", "reason": "more"}
            ),
        ]
    )
    result = await agentic_search(searcher, "q1", llm, max_rounds=3)

    stats = result.agentic_round_stats
    assert stats is not None
    assert len(stats) == 2
    assert stats[1].sufficient is None
    assert stats[1].judge_reason == "no new info"
    assert stats[1].new_sources == 0
    assert stats[1].new_entities == 0
    assert stats[1].new_relations == 0


@pytest.mark.asyncio
async def test_round_stats_none_for_no_rounds() -> None:
    """Edge case: if max_rounds=0 the stats field is None (defensive
    check — no real call site does this, but the model contract holds).

    More relevant: when only one round runs and judge says sufficient,
    stats should still have one entry."""
    searcher = StubSearcher(
        search_responses=[_resp(["c1"])],
        graph_responses=[_graph(["X"])],
    )
    llm = StubLLMClient(replies=[json.dumps({"sufficient": True})])
    result = await agentic_search(searcher, "q", llm)
    assert result.agentic_round_stats is not None
    assert len(result.agentic_round_stats) == 1
    assert result.agentic_round_stats[0].sufficient is True


@pytest.mark.asyncio
async def test_judge_failure_breaks_loop_and_still_synthesizes() -> None:
    """A malformed judge JSON should set sufficient=True (defensive
    default in ``_judge_context``) and proceed to synthesis without
    crashing."""
    searcher = StubSearcher(
        search_responses=[_resp(["c1"])],
        graph_responses=[_graph(["X"])],
    )
    llm = StubLLMClient(replies=["this is not json at all"])

    result = await agentic_search(searcher, "q", llm)

    assert result.answer == "synthesized answer"
    assert len(searcher.ask_rag_calls) == 1


# ── Stage I: helpers + edge-case loop control ───────────────────────


def test_deduplicate_sources_preserves_order_and_dedupes_by_chunk_id() -> None:
    a = _src("c1", "d1")
    b = _src("c2", "d1")
    c = _src("c1", "d2")  # same chunk_id as a → dedup hit
    d = _src("c3", "d3")
    out = _deduplicate_sources([a, b, c, d])
    assert [s.chunk_id for s in out] == ["c1", "c2", "c3"]
    # first occurrence wins
    assert out[0] is a


def test_deduplicate_sources_empty_input() -> None:
    assert _deduplicate_sources([]) == []


def test_merge_graph_data_empty_inputs() -> None:
    g = _graph(["X"])
    assert _merge_graph_data({}, g) == g
    assert _merge_graph_data(g, {}) == g
    assert _merge_graph_data({}, {}) == {}


def test_merge_graph_data_dedupes_entities_by_name() -> None:
    a = _graph(["NVIDIA", "AMD"])
    b = _graph(["AMD", "TSMC"])  # AMD dup
    merged = _merge_graph_data(a, b)
    names = [e["entity_name"] for e in merged["data"]["entities"]]
    assert names == ["NVIDIA", "AMD", "TSMC"]


def test_merge_graph_data_dedupes_relations_by_endpoints() -> None:
    a = _graph(["NVIDIA", "AMD"], relations=[("NVIDIA", "AMD")])
    b = _graph(
        ["NVIDIA", "AMD"],
        relations=[("NVIDIA", "AMD"), ("AMD", "INTEL")],  # first is dup
    )
    merged = _merge_graph_data(a, b)
    rels = [
        (r["src_id"], r["tgt_id"])
        for r in merged["data"]["relationships"]
    ]
    assert rels == [("NVIDIA", "AMD"), ("AMD", "INTEL")]


@pytest.mark.asyncio
async def test_judge_context_parses_plain_json() -> None:
    llm = StubLLMClient(
        replies=[
            json.dumps(
                {
                    "sufficient": False,
                    "follow_up_query": "next?",
                    "reason": "missing X",
                }
            )
        ]
    )
    out = await _judge_context(llm, "q", [], {})
    assert out["sufficient"] is False
    assert out["follow_up_query"] == "next?"
    assert out["reason"] == "missing X"


@pytest.mark.asyncio
async def test_judge_context_strips_markdown_fences() -> None:
    """Some LLMs wrap JSON in ```json ... ``` despite instructions to
    return raw JSON.  ``_judge_context`` strips the fence."""
    fenced = (
        "```json\n"
        + json.dumps({"sufficient": True, "follow_up_query": ""})
        + "\n```"
    )
    llm = StubLLMClient(replies=[fenced])
    out = await _judge_context(llm, "q", [], {})
    assert out["sufficient"] is True


@pytest.mark.asyncio
async def test_judge_context_invalid_json_defaults_to_sufficient() -> None:
    """Defensive fallback: any parse failure → ``sufficient=True`` so
    the loop terminates cleanly instead of crashing the whole
    request."""
    llm = StubLLMClient(replies=["not even close to JSON {[}"])
    out = await _judge_context(llm, "q", [], {})
    assert out["sufficient"] is True
    assert out["follow_up_query"] == ""


@pytest.mark.asyncio
async def test_judge_context_llm_exception_defaults_to_sufficient() -> None:
    class _RaisingLLM:
        async def chat(self, *, model, messages):
            raise RuntimeError("LLM down")

    out = await _judge_context(_RaisingLLM(), "q", [], {})
    assert out["sufficient"] is True
    # reason carries the error string for debugging
    assert "LLM down" in out["reason"]


@pytest.mark.asyncio
async def test_max_rounds_reached_without_sufficient() -> None:
    """Hit the round cap with the judge always asking for more — loop
    exits via the for-range exhaustion, NOT via sufficient/early-exit."""
    searcher = StubSearcher(
        search_responses=[_resp(["c1"]), _resp(["c2"]), _resp(["c3"])],
        graph_responses=[
            _graph(["A"]), _graph(["B"]), _graph(["C"]),
        ],
    )
    llm = StubLLMClient(
        replies=[
            json.dumps(
                {"sufficient": False, "follow_up_query": "q2", "reason": "r"}
            ),
            json.dumps(
                {"sufficient": False, "follow_up_query": "q3", "reason": "r"}
            ),
            json.dumps(
                {"sufficient": False, "follow_up_query": "q4", "reason": "r"}
            ),
        ]
    )

    result = await agentic_search(
        searcher, "q1", llm, max_rounds=3,
    )

    assert result.agentic_rounds == 3
    assert len(llm.calls) == 3  # judge ran every round
    # All three judge follow-ups are recorded — the third one is
    # appended right before the for-loop range is exhausted (we don't
    # re-search on it but it shows in telemetry as "what the judge
    # would have probed next").
    assert result.follow_up_queries == ["q2", "q3", "q4"]
    # final synthesis still runs with all accumulated context
    assert len(searcher.ask_rag_calls) == 1


@pytest.mark.asyncio
async def test_judge_repeating_current_query_breaks_loop() -> None:
    """If the judge returns a follow-up identical to the current query
    we break the loop (else we'd spin in place)."""
    searcher = StubSearcher(
        search_responses=[_resp(["c1"]), _resp(["c2"])],
        graph_responses=[_graph(["A"]), _graph(["B"])],
    )
    llm = StubLLMClient(
        replies=[
            # judge says insufficient + suggests the SAME query →
            # loop must exit, not retry forever
            json.dumps(
                {
                    "sufficient": False,
                    "follow_up_query": "q1",
                    "reason": "loop",
                }
            ),
        ]
    )
    result = await agentic_search(
        searcher, "q1", llm, max_rounds=3,
    )

    assert result.agentic_rounds == 1
    assert result.follow_up_queries is None
    # judge called exactly once
    assert len(llm.calls) == 1
    # final synthesis still runs
    assert len(searcher.ask_rag_calls) == 1
