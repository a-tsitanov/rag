"""Enterprise Knowledge Base — FastAPI app.

Run with::

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Every client (Milvus, Neo4j, Postgres, LightRAG, HybridSearcher) is
created by :mod:`src.di.providers` and injected into routes via
``FromDishka[T]``.  The RabbitMQ broker lives as a module singleton in
:mod:`src.ingestion.tasks`; we start/stop it here so the API process
can kick tasks without running any task body itself.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dishka.integrations.fastapi import setup_dishka

from src.api.middleware import RequestLoggingMiddleware, configure_logging
from src.api.routes import health, ingest, search
from src.config import settings
from src.di import build_api_container
from src.ingestion.tasks import broker

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # API is a pure kicker — never runs task bodies — so skip the
    # startup dance if we happen to be imported from `taskiq worker`.
    if not broker.is_worker_process:
        await broker.startup()
        logger.info("taskiq broker connected: %s", settings.rabbitmq.url)
    try:
        yield
    finally:
        if not broker.is_worker_process:
            await broker.shutdown()
        # dishka's setup_dishka attaches the container to app.state.dishka_container
        await app.state.dishka_container.close()


app = FastAPI(
    title="Enterprise Knowledge Base",
    description=(
        "Hybrid RAG API for ingesting enterprise documents and answering "
        "natural-language questions over them.\n\n"
        "### Authentication\n"
        "All `/api/v1/*` endpoints require an `X-API-Key` header. Keys "
        "are managed via the `API_KEYS` environment variable.\n\n"
        "### Ingestion\n"
        "`POST /api/v1/ingest` uploads a file and enqueues it on "
        "RabbitMQ.  Poll `GET /api/v1/ingest/{job_id}` for progress.\n\n"
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

# ── middleware ────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# ── DI: build container and wire into FastAPI ─────────────────────────

setup_dishka(build_api_container(), app)

# ── routes ────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(search.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
