import asyncio
import logging
import signal

from src.config import settings
from src.ingestion.pipeline import ingestion_pipeline
from src.services.milvus import milvus_service
from src.services.neo4j import neo4j_service
from src.services.redis import redis_service

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

shutdown = False


def handle_signal(sig, frame):
    global shutdown
    logger.info("Shutdown signal received")
    shutdown = True


async def run_worker():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("Connecting to services...")
    await redis_service.connect()
    await milvus_service.connect()
    neo4j_service.connect()

    logger.info("Worker started, waiting for tasks...")

    try:
        while not shutdown:
            task = await redis_service.dequeue_task()
            if task is None:
                continue

            task_id = task["task_id"]
            logger.info(f"Processing task {task_id}: {task['filename']}")

            try:
                await redis_service.set_task_status(task_id, "processing")
                result = await ingestion_pipeline.process(
                    document_id=task_id,
                    filename=task["filename"],
                    content=task["content"],
                )
                await redis_service.set_task_status(task_id, "completed")
                logger.info(f"Task {task_id} completed: {result['chunks']} chunks")
            except Exception:
                logger.exception(f"Task {task_id} failed")
                await redis_service.set_task_status(task_id, "failed")
    finally:
        await redis_service.disconnect()
        await milvus_service.disconnect()
        neo4j_service.disconnect()
        logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(run_worker())
