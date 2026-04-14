"""Singleton LightRAG instance wired to the project's storage backends.

LightRAG v1.0.0 API
--------------------
* ``kg``                                  — graph storage name (str), looked up in an
                                            internal dict; ``"Neo4JStorage"`` uses env-vars
                                            ``NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD``.
* ``vector_db_storage_cls``               — vector storage **class** (not string).
* ``key_string_value_json_storage_cls``    — KV storage **class**.
* ``llm_model_func``                      — async callable; receives model name through
                                            ``global_config["llm_model_name"]``.
* ``embedding_func``                      — ``EmbeddingFunc(embedding_dim, max_token_size, func)``.

This module exposes:

* ``init_rag()``  — call once at app startup (inside FastAPI lifespan).
* ``close_rag()`` — call once at shutdown.
* ``get_rag()``   — FastAPI dependency that returns the singleton.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

# ── singleton state ───────────────────────────────────────────────────

_rag_instance = None  # set by init_rag(), cleared by close_rag()


# ── helpers: push our config into env-vars that LightRAG reads ────────


def _export_neo4j_env() -> None:
    """Neo4JStorage reads NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD
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


async def init_rag(
    *,
    working_dir: str | None = None,
    llm_model_name: str = "llama3.3:70b",
    embed_model: str = "bge-m3:latest",
    embedding_dim: int = 1024,
    graph_storage: str = "Neo4JStorage",
) -> None:
    """Create the singleton :class:`LightRAG` and store it in module state.

    Must be called exactly once — typically in the FastAPI lifespan.
    """
    global _rag_instance

    if _rag_instance is not None:
        logger.warning("LightRAG already initialised — skipping")
        return

    # lazy import so the module is loadable without torch installed
    from lightrag import LightRAG
    from lightrag.llm import ollama_model_complete, ollama_embedding
    from lightrag.utils import EmbeddingFunc

    _export_neo4j_env()
    _export_milvus_env()

    wd = working_dir or settings.lightrag_working_dir
    Path(wd).mkdir(parents=True, exist_ok=True)

    # ── embedding wrapper ─────────────────────────────────────────
    async def _embed(texts: list[str]) -> list:
        return await ollama_embedding(texts, embed_model=embed_model)

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=_embed,
    )

    # ── build instance ────────────────────────────────────────────
    _rag_instance = LightRAG(
        working_dir=wd,
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model_name,
        embedding_func=embedding_func,
        kg=graph_storage,
    )

    logger.info(
        "LightRAG initialised  working_dir=%s  llm=%s  embed=%s  graph=%s",
        wd, llm_model_name, embed_model, graph_storage,
    )


async def close_rag() -> None:
    """Shut down the singleton (close graph driver, flush caches)."""
    global _rag_instance
    if _rag_instance is None:
        return

    try:
        graph = _rag_instance.chunk_entity_relation_graph
        if hasattr(graph, "close"):
            await graph.close()
    except Exception as exc:
        logger.warning("Error closing graph storage: %s", exc)

    _rag_instance = None
    logger.info("LightRAG shut down")


# ── FastAPI dependency ────────────────────────────────────────────────


def get_rag():
    """FastAPI ``Depends(get_rag)`` — returns the singleton LightRAG."""
    if _rag_instance is None:
        raise RuntimeError(
            "LightRAG not initialised. "
            "Call init_rag() in your FastAPI lifespan first."
        )
    return _rag_instance
