import json

import redis.asyncio as aioredis

from src.config import settings


class RedisService:
    def __init__(self):
        self.client: aioredis.Redis | None = None

    async def connect(self):
        self.client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    async def disconnect(self):
        if self.client:
            await self.client.close()

    async def enqueue_task(self, task_id: str, payload: dict):
        await self.client.lpush(
            settings.ingestion_queue,
            json.dumps({"task_id": task_id, **payload}),
        )
        await self.client.set(f"task:{task_id}:status", "queued")

    async def dequeue_task(self) -> dict | None:
        result = await self.client.brpop(settings.ingestion_queue, timeout=5)
        if result:
            return json.loads(result[1])
        return None

    async def set_task_status(self, task_id: str, status: str):
        await self.client.set(f"task:{task_id}:status", status)

    async def get_task_status(self, task_id: str) -> str | None:
        return await self.client.get(f"task:{task_id}:status")


redis_service = RedisService()
