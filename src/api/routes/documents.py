import uuid

from fastapi import APIRouter, Depends, UploadFile

from src.api.dependencies import get_redis
from src.models.document import DocumentResponse, IngestResponse
from src.services.redis import RedisService

router = APIRouter(tags=["documents"])


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile,
    redis: RedisService = Depends(get_redis),
):
    task_id = str(uuid.uuid4())
    content = await file.read()

    await redis.enqueue_task(
        task_id=task_id,
        payload={
            "filename": file.filename,
            "content_type": file.content_type,
            "content": content.decode("utf-8", errors="replace"),
        },
    )

    return IngestResponse(task_id=task_id, status="queued")


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    # TODO: implement document retrieval from stores
    return DocumentResponse(
        id=document_id,
        filename="",
        status="not_implemented",
        chunk_count=0,
    )
