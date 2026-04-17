"""Health endpoint — pings every backend."""

from __future__ import annotations

import asyncio
from typing import Literal

import aio_pika
import ollama
import psycopg
from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter
from lightrag import LightRAG
from pydantic import BaseModel, Field

from src.config import Settings
from src.storage.milvus_client import check_milvus_health
from src.storage.neo4j_client import AsyncNeo4jClient

router = APIRouter(tags=["health"])


class ServiceHealth(BaseModel):
    status: Literal["up", "down"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="healthy=все up, degraded=часть down, unhealthy=все down",
    )
    services: dict[str, ServiceHealth]


# ── per-service checks ───────────────────────────────────────────────


async def _check_rabbitmq(s: Settings) -> ServiceHealth:
    try:
        conn = await asyncio.wait_for(
            aio_pika.connect(s.rabbitmq.url), timeout=s.rabbitmq.timeout_s,
        )
        try:
            return ServiceHealth(status="up")
        finally:
            await conn.close()
    except Exception as exc:
        return ServiceHealth(status="down", detail=str(exc))


async def _check_milvus(s: Settings) -> ServiceHealth:
    try:
        ver = await check_milvus_health()
        return ServiceHealth(status="up", detail=str(ver))
    except Exception as exc:
        return ServiceHealth(status="down", detail=str(exc))


async def _check_neo4j(n: AsyncNeo4jClient) -> ServiceHealth:
    try:
        await n._driver.verify_connectivity()
        return ServiceHealth(status="up")
    except Exception as exc:
        return ServiceHealth(status="down", detail=str(exc))


async def _check_postgres(
    pg: psycopg.AsyncConnection, s: Settings,
) -> ServiceHealth:
    try:
        cur = await asyncio.wait_for(
            pg.execute("SELECT version()"), timeout=s.postgres.connect_timeout_s,
        )
        row = await cur.fetchone()
        return ServiceHealth(
            status="up",
            detail=(row[0].split(",")[0] if row else None),
        )
    except Exception as exc:
        return ServiceHealth(status="down", detail=str(exc))


async def _check_ollama(
    oc: ollama.AsyncClient, s: Settings,
) -> ServiceHealth:
    try:
        resp = await asyncio.wait_for(oc.list(), timeout=min(s.ollama.timeout_s, 5.0))
        models = [m.model for m in (resp.models or [])]
        return ServiceHealth(status="up", detail=f"{len(models)} models")
    except Exception as exc:
        return ServiceHealth(status="down", detail=str(exc))


async def _check_lightrag(rag: LightRAG) -> ServiceHealth:
    try:
        assert hasattr(rag, "ainsert")
        return ServiceHealth(status="up")
    except Exception as exc:
        return ServiceHealth(status="down", detail=str(exc))


# ── endpoint ──────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness + dependency health",
)
@inject
async def health(
    settings: FromDishka[Settings],
    neo4j: FromDishka[AsyncNeo4jClient],
    pg: FromDishka[psycopg.AsyncConnection],
    ollama_client: FromDishka[ollama.AsyncClient],
    rag: FromDishka[LightRAG],
) -> HealthResponse:
    rmq, m, n, p, o, rg = await asyncio.gather(
        _check_rabbitmq(settings),
        _check_milvus(settings),
        _check_neo4j(neo4j),
        _check_postgres(pg, settings),
        _check_ollama(ollama_client, settings),
        _check_lightrag(rag),
    )

    services = {
        "rabbitmq": rmq, "milvus": m, "neo4j": n,
        "postgres": p, "ollama": o, "lightrag": rg,
    }

    down = [s for s in services.values() if s.status == "down"]
    if not down:
        overall = "healthy"
    elif len(down) == len(services):
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthResponse(status=overall, services=services)
