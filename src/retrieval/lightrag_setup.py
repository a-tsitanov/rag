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
    os.environ.setdefault("NEO4J_URI", settings.neo4j_uri)
    os.environ.setdefault("NEO4J_USERNAME", settings.neo4j_user)
    os.environ.setdefault("NEO4J_PASSWORD", settings.neo4j_password)


def _export_milvus_env() -> None:
    os.environ.setdefault(
        "MILVUS_URI",
        f"http://{settings.milvus_host}:{settings.milvus_port}",
    )


# ── factory ───────────────────────────────────────────────────────────


async def create_rag(
    *,
    working_dir: str | None = None,
    llm_model_name: str = "llama3.3:70b",
    embed_model: str = "bge-m3:latest",
    embedding_dim: int = 1024,
    graph_storage: str | None = None,
):
    """Build and return a fresh LightRAG instance.

    Caller owns teardown.  For the Neo4j backend call
    :func:`close_rag_graph` on the returned instance to close the driver.
    """
    # Lazy imports so this module is importable without torch.
    from lightrag import LightRAG
    from lightrag.llm import ollama_embedding, ollama_model_complete
    from lightrag.utils import EmbeddingFunc

    _export_neo4j_env()
    _export_milvus_env()

    wd = working_dir or settings.lightrag_working_dir
    Path(wd).mkdir(parents=True, exist_ok=True)

    async def _embed(texts: list[str]) -> list:
        return await ollama_embedding(texts, embed_model=embed_model)

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=_embed,
    )

    rag = LightRAG(
        working_dir=wd,
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model_name,
        embedding_func=embedding_func,
        graph_storage=graph_storage or settings.lightrag_graph_storage,
    )

    # LightRAG 1.4+ requires explicit storage initialization before any
    # ainsert / aquery call.  It populates pipeline_status and opens the
    # graph driver.
    await rag.initialize_storages()

    logger.info(
        "LightRAG created  working_dir=%s  llm=%s  embed=%s  graph=%s",
        wd, llm_model_name, embed_model,
        graph_storage or settings.lightrag_graph_storage,
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
