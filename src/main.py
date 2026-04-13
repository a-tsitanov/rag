from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import documents, health, search
from src.services.milvus import milvus_service
from src.services.neo4j import neo4j_service
from src.services.redis import redis_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    await redis_service.connect()
    await milvus_service.connect()
    neo4j_service.connect()
    yield
    await redis_service.disconnect()
    await milvus_service.disconnect()
    neo4j_service.disconnect()


app = FastAPI(
    title="Enterprise Knowledge Base",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(documents.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
