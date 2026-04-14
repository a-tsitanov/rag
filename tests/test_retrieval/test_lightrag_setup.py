"""Integration test for LightRAG setup.

Uses NetworkX graph storage (in-process, no Neo4j needed) and stub
LLM / embedding functions so the test runs without Ollama.

The test inserts a document and verifies that LightRAG extracted
entities into its graph storage.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

# ── skip the entire module if lightrag can't import (needs torch) ─────

pytest.importorskip("lightrag", reason="lightrag (+ torch) not installed")

import src.retrieval.lightrag_setup as rag_mod  # noqa: E402
from lightrag import LightRAG  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402


# ── stub LLM: returns tuples in the format LightRAG's operate.py expects
#
#    Format:  ("entity"<|>NAME<|>TYPE<|>DESCRIPTION)##
#             ("relationship"<|>SRC<|>TGT<|>DESC<|>KEYWORDS<|>WEIGHT)##
#             <|COMPLETE|>

_TUPLE_SEP = "<|>"
_RECORD_SEP = "##"
_DONE = "<|COMPLETE|>"

_ENTITY_RESPONSE = (
    f'("entity"{_TUPLE_SEP}"ACME CORP"{_TUPLE_SEP}"ORGANIZATION"'
    f'{_TUPLE_SEP}"A technology company"){_RECORD_SEP}'
    f'("entity"{_TUPLE_SEP}"BOB SMITH"{_TUPLE_SEP}"PERSON"'
    f'{_TUPLE_SEP}"CTO of Acme Corp"){_RECORD_SEP}'
    f'("entity"{_TUPLE_SEP}"KUBERNETES"{_TUPLE_SEP}"TECHNOLOGY"'
    f'{_TUPLE_SEP}"Container orchestration platform"){_RECORD_SEP}'
    f'("relationship"{_TUPLE_SEP}"BOB SMITH"{_TUPLE_SEP}"ACME CORP"'
    f'{_TUPLE_SEP}"Bob Smith is CTO of Acme Corp"'
    f'{_TUPLE_SEP}"leadership"{_TUPLE_SEP}1.0){_RECORD_SEP}'
    f'("relationship"{_TUPLE_SEP}"ACME CORP"{_TUPLE_SEP}"KUBERNETES"'
    f'{_TUPLE_SEP}"Acme Corp uses Kubernetes"'
    f'{_TUPLE_SEP}"technology"{_TUPLE_SEP}0.8){_RECORD_SEP}'
    f"{_DONE}"
)


async def _stub_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
    """Return pre-canned entity-extraction tuples, or 'YES' for glean loop."""
    if "YES" in prompt or "no" in prompt.lower():
        return "YES"
    return _ENTITY_RESPONSE


async def _stub_embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) & 0xFFFF_FFFF)
        v = rng.randn(384).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-10
        vecs.append(v)
    return np.array(vecs, dtype=np.float32)


# ── fixtures ──────────────────────────────────────────────────────────

WORKING_DIR = Path("/tmp/lightrag_test_workdir")


@pytest_asyncio.fixture
async def rag():
    """Throwaway LightRAG with NetworkX graph + stub LLM."""
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    embedding_func = EmbeddingFunc(
        embedding_dim=384,
        max_token_size=8192,
        func=_stub_embed,
    )

    instance = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=_stub_llm,
        llm_model_name="stub",
        embedding_func=embedding_func,
        kg="NetworkXStorage",
        enable_llm_cache=False,
    )

    yield instance

    shutil.rmtree(WORKING_DIR, ignore_errors=True)


TEST_DOCUMENT = textwrap.dedent("""\
    Acme Corp is a technology company founded in 2020.
    Bob Smith serves as the CTO and leads the engineering team.
    The company uses Kubernetes for container orchestration
    and deploys microservices across multiple cloud regions.
    Their knowledge base system processes thousands of documents daily.
""")


# ── entity extraction ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_insert_document_populates_graph(rag: LightRAG):
    await rag.ainsert(TEST_DOCUMENT)

    graph = rag.chunk_entity_relation_graph
    nx_graph = graph._graph

    nodes = list(nx_graph.nodes())
    edges = list(nx_graph.edges())

    print(f"\n  Graph nodes ({len(nodes)}): {nodes}")
    print(f"  Graph edges ({len(edges)}): {edges}")

    assert len(nodes) >= 2, f"Expected entities, got: {nodes}"
    assert len(edges) >= 1, f"Expected relationships, got: {edges}"


@pytest.mark.asyncio
async def test_insert_stores_full_doc_and_chunks(rag: LightRAG):
    await rag.ainsert(TEST_DOCUMENT)

    doc_keys = await rag.full_docs.all_keys()
    chunk_keys = await rag.text_chunks.all_keys()

    print(f"\n  full_docs:   {len(doc_keys)} entries")
    print(f"  text_chunks: {len(chunk_keys)} entries")

    assert len(doc_keys) >= 1
    assert len(chunk_keys) >= 1


# ── singleton lifecycle ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_rag_before_init_raises():
    rag_mod._rag_instance = None
    with pytest.raises(RuntimeError, match="not initialised"):
        rag_mod.get_rag()


@pytest.mark.asyncio
async def test_init_and_close_lifecycle():
    wd = "/tmp/lightrag_lifecycle_test"
    if Path(wd).exists():
        shutil.rmtree(wd)
    Path(wd).mkdir(parents=True, exist_ok=True)

    rag_mod._rag_instance = None

    try:
        embedding_func = EmbeddingFunc(
            embedding_dim=384, max_token_size=8192, func=_stub_embed,
        )

        instance = LightRAG(
            working_dir=wd,
            llm_model_func=_stub_llm,
            llm_model_name="stub",
            embedding_func=embedding_func,
            kg="NetworkXStorage",
            enable_llm_cache=False,
        )
        rag_mod._rag_instance = instance

        r = rag_mod.get_rag()
        assert r is instance

        await rag_mod.close_rag()
        assert rag_mod._rag_instance is None
    finally:
        rag_mod._rag_instance = None
        shutil.rmtree(wd, ignore_errors=True)
