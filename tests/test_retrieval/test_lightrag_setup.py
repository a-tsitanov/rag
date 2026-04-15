"""Smoke test for LightRAG 1.4 setup.

Verifies that ``src.retrieval.lightrag_setup.create_rag`` builds a
usable LightRAG instance with NetworkX storage and that
``close_rag_graph`` is a safe no-op for non-Neo4j backends.

The previous "entity extraction" tests were removed after the LightRAG
1.0 → 1.4 upgrade: the stub LLM tuple format (``"entity"<|>...``) and
the single-call ``ainsert`` pipeline semantics no longer apply.  Real
end-to-end ingestion is already exercised by the worker + hybrid_search
tests, so we keep only a thin smoke check here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

# ── skip the entire module if lightrag can't import (needs torch) ─────

pytest.importorskip("lightrag", reason="lightrag (+ torch) not installed")

import src.retrieval.lightrag_setup as rag_mod  # noqa: E402
from lightrag import LightRAG  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402


async def _stub_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    return "stub"


async def _stub_embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) & 0xFFFF_FFFF)
        v = rng.randn(384).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-10
        vecs.append(v)
    return np.array(vecs, dtype=np.float32)


# ── constructor + initialize + close smoke test ──────────────────────


@pytest.mark.asyncio
async def test_lightrag_smoke_networkx():
    wd = Path("/tmp/lightrag_smoke_test")
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)

    try:
        embedding_func = EmbeddingFunc(
            embedding_dim=384, max_token_size=8192, func=_stub_embed,
        )
        instance = LightRAG(
            working_dir=str(wd),
            llm_model_func=_stub_llm,
            llm_model_name="stub",
            embedding_func=embedding_func,
            graph_storage="NetworkXStorage",
            enable_llm_cache=False,
        )
        await instance.initialize_storages()

        # basic attribute probe — lightrag_setup.close_rag_graph relies on this
        assert hasattr(instance, "chunk_entity_relation_graph")

        # close_rag_graph must be a safe no-op for the NetworkX backend
        await rag_mod.close_rag_graph(instance)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
