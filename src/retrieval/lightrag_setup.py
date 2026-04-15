"""LightRAG factory + graph-storage env shims.

Public API is now a single pure factory :func:`create_rag` — it returns a
fresh :class:`lightrag.LightRAG` instance without touching any module
globals.  Lifecycle (teardown) is the caller's responsibility (the dishka
provider does it via ``yield`` + ``close_rag_graph``).

The old singleton triad (``init_rag`` / ``get_rag`` / ``close_rag``) was
removed — a container now owns the instance.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


# ── env-var bridges for LightRAG storage backends ────────────────────


def _export_neo4j_env() -> None:
    """``Neo4JStorage`` reads ``NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD``
    from the process environment — not from constructor args."""
    os.environ.setdefault("NEO4J_URI", settings.neo4j.uri)
    os.environ.setdefault("NEO4J_USERNAME", settings.neo4j.user)
    os.environ.setdefault("NEO4J_PASSWORD", settings.neo4j.password)


def _export_milvus_env() -> None:
    os.environ.setdefault(
        "MILVUS_URI",
        f"http://{settings.milvus.host}:{settings.milvus.port}",
    )


# ── factory ───────────────────────────────────────────────────────────


async def create_rag(
    *,
    working_dir: str | None = None,
    llm_model_name: str | None = None,
    embed_model: str | None = None,
    embedding_dim: int | None = None,
    max_token_size: int | None = None,
    graph_storage: str | None = None,
):
    """Build and return a fresh LightRAG instance.

    All model/dim args default to the resolved values in
    :class:`~src.config.Settings` — set
    ``LIGHTRAG_LLM_MODEL`` / ``LIGHTRAG_EMBEDDING_MODEL`` /
    ``LIGHTRAG_EMBEDDING_DIM`` in the environment to override, or leave
    them empty to reuse ``OLLAMA_MODEL`` / ``EMBEDDING_MODEL`` /
    ``EMBEDDING_DIM``.

    Caller owns teardown.  For the Neo4j backend call
    :func:`close_rag_graph` on the returned instance to close the driver.
    """
    # Lazy imports so this module is importable without torch.
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
    from lightrag.utils import EmbeddingFunc

    _export_neo4j_env()
    _export_milvus_env()

    wd = working_dir or settings.lightrag.working_dir
    llm_name = llm_model_name or settings.effective_lightrag_llm_model
    embed_name = embed_model or settings.effective_lightrag_embedding_model
    embed_dim = embedding_dim or settings.effective_lightrag_embedding_dim
    max_tok = max_token_size or settings.lightrag.max_token_size
    graph_kind = graph_storage or settings.lightrag.graph_storage

    Path(wd).mkdir(parents=True, exist_ok=True)

    async def _embed(texts: list[str]) -> list:
        return await ollama_embed(texts, embed_model=embed_name)

    embedding_func = EmbeddingFunc(
        embedding_dim=embed_dim,
        max_token_size=max_tok,
        func=_embed,
    )

    rag = LightRAG(
        working_dir=wd,
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_name,
        embedding_func=embedding_func,
        graph_storage=graph_kind,
    )

    # LightRAG 1.4+ requires explicit storage initialization before any
    # ainsert / aquery call.  It populates pipeline_status and opens the
    # graph driver.
    await rag.initialize_storages()

    logger.info(
        "LightRAG created  working_dir=%s  llm=%s  embed=%s  dim=%d  "
        "max_tokens=%d  graph=%s",
        wd, llm_name, embed_name, embed_dim, max_tok, graph_kind,
    )
    return rag


async def close_rag_graph(rag) -> None:
    """Best-effort shutdown of LightRAG's graph driver (Neo4j).

    NetworkX graphs write to disk on ``index_done_callback`` — no close
    needed. Neo4j keeps a driver open that must be closed explicitly.
    """
    try:
        graph = getattr(rag, "chunk_entity_relation_graph", None)
        if graph is not None and hasattr(graph, "close"):
            await graph.close()
    except Exception as exc:
        logger.warning("error closing LightRAG graph: %s", exc)
