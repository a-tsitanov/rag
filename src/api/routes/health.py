"""Health endpoint — pings every backend."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

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


async def _check_redis(r) -> ServiceHealth:
    if r is None:
        return ServiceHealth(status="down", detail="not initialised")
    try:
        await r.ping()
        info = await r.info("server")
        ver = info.get("redis_version") if isinstance(info, dict) else None
        return ServiceHealth(status="up", detail=f"v{ver}" if ver else None)
    except Exception as exc:
        return ServiceHealth(status="down", detail=str(exc))


async def _check_milvus(m) -> ServiceHealth:
    if m is None:
        return ServiceHealth(status="down", detail="not initialised")
    try:
        version = await asyncio.to_thread(m._client.get_server_version)
        return ServiceHealth(status="up", detail=str(version))
    except Exception as exc:
        return ServiceHealth(status="down", detail=str(exc))


async def _check_neo4j(n) -> ServiceHealth:
    if n is None:
        return ServiceHealth(status="down", detail="not initialised")
    try:
        await n._driver.verify_connectivity()
        return ServiceHealth(status="up")
    except Exception as exc:
        return ServiceHealth(status="down", detail=str(exc))


async def _check_lightrag(rag) -> ServiceHealth:
    if rag is None:
        return ServiceHealth(status="down", detail="not initialised")
    try:
        # NetworkXStorage keeps `_graph`; Neo4JStorage keeps `_driver`.
        # We only validate the object exists and has an ainsert method.
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
async def health(request: Request) -> HealthResponse:
    state = request.app.state

    redis, milvus, neo4j, lightrag = await asyncio.gather(
        _check_redis(getattr(state, "redis", None)),
        _check_milvus(getattr(state, "milvus", None)),
        _check_neo4j(getattr(state, "neo4j", None)),
        _check_lightrag(getattr(state, "rag", None)),
    )

    services = {
        "redis": redis,
        "milvus": milvus,
        "neo4j": neo4j,
        "lightrag": lightrag,
    }

    down = [s for s in services.values() if s.status == "down"]
    if not down:
        overall = "healthy"
    elif len(down) == len(services):
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthResponse(status=overall, services=services)
