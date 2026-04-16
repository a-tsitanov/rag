"""Sanity-тест на /health — поднимает FastAPI через ASGITransport и
проверяет, что эндпоинт отдаёт валидный ``HealthResponse`` (не ``200 ok``,
а структуру с ``status`` и ``services``).

Реальные бэкенды не требуются: dishka DI-контейнер построится с живыми
клиентами, и если Milvus/Neo4j/Postgres/Rabbit/Ollama не подняты —
endpoint отдаст ``degraded`` / ``unhealthy``, но 200 и схема останутся
валидными.  Поэтому проверяем только форму ответа.
"""

import pytest


@pytest.mark.asyncio
async def test_health_endpoint_returns_valid_schema(client):
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded", "unhealthy"}
    assert "services" in body
    assert isinstance(body["services"], dict)
    # по одному per-service описанию с полями status/detail
    for name, svc in body["services"].items():
        assert svc["status"] in {"up", "down"}
