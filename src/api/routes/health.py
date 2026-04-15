"""Health endpoint — pings every backend."""

from __future__ import annotations

import asyncio
from typing import Literal

import aio_pika
from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter
from lightrag import LightRAG
from pydantic import BaseModel, Field

from src.config import Settings
from src.storage.milvus_client import AsyncMilvusClient
from src.storage.neo4j_client import AsyncNeo4jClient

router = APIRouter(tags=["health"])


# ── schemas ───────────────────────────────────────────────────────────


class ServiceHealth(BaseModel):
    status: Literal["up", "down"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="``healthy``: all services up, "
        "``degraded``: some down, ``unhealthy``: all down.",
    )
    services: dict[str, ServiceHealth]


# ── per-service checks ───────────────────────────────────────────────


async def _check_rabbitmq(s: Settings) -> ServiceHealth:
    try:
        conn = await asyncio.wait_for(
            aio_pika.connect(s.rabbitmq_url), timeout=3.0,
        )
        try:
            return ServiceHealth(status="up")
        finally:
            await conn.close()
    except Exception as exc:
        return ServiceHealth(status="down", detail=str(exc))


async def _check_milvus(m: AsyncMilvusClient) -> ServiceHealth:
    try:
        version = await asyncio.to_thread(m._client.get_server_version)
        return ServiceHealth(status="up", detail=str(version))
    except Exception as exc:
        return ServiceHealth(status="down", detail=str(exc))


async def _check_neo4j(n: AsyncNeo4jClient) -> ServiceHealth:
    try:
        await n._driver.verify_connectivity()
        return ServiceHealth(status="up")
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
    description=(
        "Pings every backend the API depends on and returns a per-service "
        "status.  Use this for load-balancer / k8s probes.  **Public** — no "
        "API key required."
    ),
)
@inject
async def health(
    settings: FromDishka[Settings],
    milvus: FromDishka[AsyncMilvusClient],
    neo4j: FromDishka[AsyncNeo4jClient],
    rag: FromDishka[LightRAG],
) -> HealthResponse:
    rmq, m, n, rg = await asyncio.gather(
        _check_rabbitmq(settings),
        _check_milvus(milvus),
        _check_neo4j(neo4j),
        _check_lightrag(rag),
    )

    services = {"rabbitmq": rmq, "milvus": m, "neo4j": n, "lightrag": rg}

    down = [s for s in services.values() if s.status == "down"]
    if not down:
        overall = "healthy"
    elif len(down) == len(services):
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthResponse(status=overall, services=services)
