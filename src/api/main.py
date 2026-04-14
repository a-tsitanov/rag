"""Enterprise Knowledge Base — FastAPI app.

Run with::

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
* ``GET  /health``                  — public liveness + dependency health
* ``POST /api/v1/search``           — hybrid RAG + vector search
* ``POST /api/v1/ingest``           — enqueue an uploaded document
* ``GET  /api/v1/ingest/{job_id}``  — poll ingestion progress

All ``/api/v1/*`` endpoints require ``X-API-Key`` header (see ``.env``).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import RequestLoggingMiddleware, configure_logging
from src.api.routes import health, ingest, search
from src.config import settings
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.lightrag_setup import close_rag, get_rag, init_rag
from src.storage.milvus_client import AsyncMilvusClient
from src.storage.neo4j_client import AsyncNeo4jClient

configure_logging()
logger = logging.getLogger(__name__)


# ── embed + rerank factories ──────────────────────────────────────────


def _build_ollama_embed_fn():
    """Return an async embed function that calls Ollama's embeddings API.

    Fails at call-time (not import-time) if Ollama is unreachable — the
    endpoint surfaces the error as a 500.
    """
    import ollama

    client = ollama.AsyncClient(host=settings.ollama_host)

    async def _embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            resp = await client.embeddings(
                model=settings.embedding_model, prompt=t,
            )
            vecs.append(resp["embedding"])
        return np.array(vecs, dtype=np.float32)

    return _embed


def _stub_reranker(query: str, candidates: list[str]) -> list[float]:
    """Placeholder while BGE-reranker-v2-m3 isn't pre-downloaded.

    Keeps Milvus' original order.  Swap out for a real cross-encoder
    (e.g. ``sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")``)
    when the model weights are available locally.
    """
    return [1.0 - i * 0.01 for i in range(len(candidates))]


# ── lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Redis — used by queue + cache
    app.state.redis = aioredis.from_url(
        settings.redis_url, decode_responses=False,
    )
    try:
        await app.state.redis.ping()
        logger.info("redis connected: %s", settings.redis_url)
    except Exception as exc:
        logger.warning("redis ping failed: %s", exc)

    # Milvus
    try:
        app.state.milvus = AsyncMilvusClient(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
        )
        await app.state.milvus.connect()
        logger.info("milvus connected")
    except Exception as exc:
        logger.warning("milvus connect failed: %s", exc)
        app.state.milvus = None

    # Neo4j
    try:
        app.state.neo4j = AsyncNeo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        await app.state.neo4j.connect()
        logger.info("neo4j connected: %s", settings.neo4j_uri)
    except Exception as exc:
        logger.warning("neo4j connect failed: %s", exc)
        app.state.neo4j = None

    # LightRAG (NetworkX is safer than Neo4j for local dev — switch to
    # "Neo4JStorage" when running against the full stack)
    try:
        await init_rag(graph_storage="NetworkXStorage")
        app.state.rag = get_rag()
        logger.info("lightrag initialised")
    except Exception as exc:
        logger.warning("lightrag init failed: %s", exc)
        app.state.rag = None

    # HybridSearcher — needs milvus + rag
    if app.state.milvus is not None and app.state.rag is not None:
        app.state.searcher = HybridSearcher(
            rag=app.state.rag,
            milvus=app.state.milvus,
            embed_fn=_build_ollama_embed_fn(),
            reranker_fn=_stub_reranker,
        )
        logger.info("hybrid searcher ready")
    else:
        app.state.searcher = None
        logger.warning(
            "hybrid searcher unavailable (milvus=%s, rag=%s)",
            app.state.milvus is not None, app.state.rag is not None,
        )

    yield

    # ── teardown ──────────────────────────────────────────────────
    if app.state.milvus is not None:
        await app.state.milvus.disconnect()
    if app.state.neo4j is not None:
        await app.state.neo4j.disconnect()
    if app.state.rag is not None:
        await close_rag()
    await app.state.redis.aclose()


# ── app ───────────────────────────────────────────────────────────────


app = FastAPI(
    title="Enterprise Knowledge Base",
    description=(
        "Hybrid RAG API for ingesting enterprise documents and answering "
        "natural-language questions over them.\n\n"
        "### Authentication\n"
        "All `/api/v1/*` endpoints require an `X-API-Key` header. Keys are "
        "managed via the `API_KEYS` environment variable.\n\n"
        "### Ingestion\n"
        "`POST /api/v1/ingest` uploads a file and enqueues it on a Redis "
        "stream.  Poll `GET /api/v1/ingest/{job_id}` for progress.\n\n"
        "### Search\n"
        "`POST /api/v1/search` runs: embed → Milvus vector search "
        "(department filter) → cross-encoder rerank → LightRAG answer "
        "generation in the requested mode."
    ),
    version="0.1.0",
    lifespan=lifespan,
    contact={"name": "Enterprise KB team"},
    openapi_tags=[
        {"name": "health", "description": "Liveness + backend health"},
        {"name": "search", "description": "Semantic search over the KB"},
        {"name": "ingestion", "description": "Document upload + job status"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# Public
app.include_router(health.router)

# Authenticated
app.include_router(search.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
