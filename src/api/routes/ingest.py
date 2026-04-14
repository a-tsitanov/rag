"""POST /api/v1/ingest — upload document and enqueue for processing.
GET  /api/v1/ingest/{job_id} — query job status."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from src.api.auth import require_api_key
from src.ingestion.queue import STATUS_PREFIX, DocumentProducer

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("/tmp/enterprise-kb-uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(tags=["ingestion"])


# ── schemas ───────────────────────────────────────────────────────────


class IngestEnqueuedResponse(BaseModel):
    job_id: str = Field(..., description="UUID to poll via GET /ingest/{job_id}")
    status: Literal["queued"] = "queued"
    path: str = Field(..., description="Server-side path where the upload was saved")


class JobProgressResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "done", "failed", "unknown"]
    path: str | None = None
    department: str | None = None
    priority: str | None = None
    attempt: int = 0
    error: str | None = None


# ── POST /ingest ──────────────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestEnqueuedResponse,
    dependencies=[Depends(require_api_key)],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document and enqueue it for ingestion",
    description=(
        "Saves the uploaded file to server-side storage and pushes a job "
        "onto the Redis stream ``documents:pending``.  A worker will pick "
        "it up asynchronously.  Poll `GET /ingest/{job_id}` for progress."
    ),
    responses={
        401: {"description": "Missing X-API-Key"},
        403: {"description": "Invalid X-API-Key"},
        503: {"description": "Queue backend unavailable"},
    },
)
async def ingest(
    request: Request,
    file: UploadFile = File(..., description="Document to ingest (PDF/DOCX/PPTX/TXT/MD/EML)"),
    department: str = Form("", description="Access-control group"),
    priority: Literal["low", "normal", "high"] = Form(
        "normal", description="Scheduling hint for the worker pool",
    ),
) -> IngestEnqueuedResponse:
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis unavailable",
        )

    # persist upload
    safe_name = (file.filename or "upload.bin").replace("/", "_")
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    dest.write_bytes(await file.read())
    await file.close()

    # enqueue
    producer = DocumentProducer(redis)
    job_id = await producer.add_document(
        path=str(dest),
        department=department,
        priority=priority,
    )

    logger.info(
        "ingest enqueued", extra={
            "job_id": job_id, "path": str(dest),
            "department": department, "priority": priority,
        },
    )
    return IngestEnqueuedResponse(job_id=job_id, status="queued", path=str(dest))


# ── GET /ingest/{job_id} ──────────────────────────────────────────────


def _decode(data: dict) -> dict:
    return {
        (k.decode() if isinstance(k, bytes) else k):
        (v.decode() if isinstance(v, bytes) else v)
        for k, v in data.items()
    }


@router.get(
    "/ingest/{job_id}",
    response_model=JobProgressResponse,
    dependencies=[Depends(require_api_key)],
    summary="Get ingestion job status",
    description=(
        "Reads the Redis hash ``job:{job_id}`` and returns the full "
        "progress record: current status, attempt count, last error "
        "(if failed), and original submission metadata."
    ),
    responses={
        401: {"description": "Missing X-API-Key"},
        403: {"description": "Invalid X-API-Key"},
        404: {"description": "Job ID unknown"},
        503: {"description": "Redis unavailable"},
    },
)
async def job_status(job_id: str, request: Request) -> JobProgressResponse:
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis unavailable",
        )

    raw = await redis.hgetall(f"{STATUS_PREFIX}{job_id}")
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    d = _decode(raw)

    return JobProgressResponse(
        job_id=job_id,
        status=d.get("status", "unknown"),  # type: ignore[arg-type]
        path=d.get("path"),
        department=d.get("department"),
        priority=d.get("priority"),
        attempt=int(d.get("attempt", 0)),
        error=d.get("error"),
    )
