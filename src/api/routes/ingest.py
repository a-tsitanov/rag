"""POST /api/v1/ingest — upload document and enqueue for processing.
GET  /api/v1/ingest/{job_id} — query job status from Postgres."""

import logging
import uuid
from pathlib import Path
from typing import Literal

import psycopg
from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.api.auth import require_api_key
from src.ingestion.tasks import priority_value, process_document
from src.ingestion.worker import PGStatusUpdater

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
    status: Literal["pending", "processing", "completed", "failed", "unknown"]
    path: str | None = None
    department: str | None = None
    error: str | None = None


# ── POST /ingest ──────────────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestEnqueuedResponse,
    dependencies=[Depends(require_api_key)],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document and enqueue it for ingestion",
    description=(
        "Saves the uploaded file to server-side storage, inserts a "
        "``documents`` row with ``status='pending'``, and kicks a "
        "``process_document`` task onto RabbitMQ.  A taskiq worker will "
        "pick it up asynchronously.  Poll ``GET /ingest/{job_id}`` for "
        "progress."
    ),
    responses={
        401: {"description": "Missing X-API-Key"},
        403: {"description": "Invalid X-API-Key"},
    },
)
@inject
async def ingest(
    pg_status: FromDishka[PGStatusUpdater],
    file: UploadFile = File(..., description="Document to ingest (PDF/DOCX/PPTX/TXT/MD/EML)"),
    department: str = Form("", description="Access-control group"),
    priority: Literal["low", "normal", "high"] = Form(
        "normal", description="Scheduling hint for the worker pool",
    ),
) -> IngestEnqueuedResponse:
    # persist upload
    safe_name = (file.filename or "upload.bin").replace("/", "_")
    doc_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{doc_id}_{safe_name}"
    dest.write_bytes(await file.read())
    await file.close()

    # insert pending row so GET /ingest/{id} has something to read
    # even before the worker wakes up
    await pg_status(
        doc_id=doc_id,
        status="pending",
        path=str(dest),
        department=department,
        error="",
    )

    # kick the task with priority label
    await (
        process_document.kicker()
        .with_labels(priority=priority_value(priority))
        .kiq(
            doc_id=doc_id,
            path=str(dest),
            department=department,
            priority=priority,
        )
    )

    logger.info(
        "ingest enqueued",
        extra={
            "job_id": doc_id, "path": str(dest),
            "department": department, "priority": priority,
        },
    )
    return IngestEnqueuedResponse(job_id=doc_id, status="queued", path=str(dest))


# ── GET /ingest/{job_id} ──────────────────────────────────────────────


_SELECT_DOC = """
    SELECT status, path, department, error
    FROM documents
    WHERE id = %s::uuid
"""


@router.get(
    "/ingest/{job_id}",
    response_model=JobProgressResponse,
    dependencies=[Depends(require_api_key)],
    summary="Get ingestion job status",
    description=(
        "Reads the ``documents`` row in Postgres and returns the current "
        "status (``pending`` → ``processing`` → ``completed`` | "
        "``failed``), the original path, the department, and the error "
        "string (if failed)."
    ),
    responses={
        401: {"description": "Missing X-API-Key"},
        403: {"description": "Invalid X-API-Key"},
        404: {"description": "Job ID unknown"},
    },
)
@inject
async def job_status(
    job_id: str,
    pg: FromDishka[psycopg.AsyncConnection],
) -> JobProgressResponse:
    try:
        cur = await pg.execute(_SELECT_DOC, (job_id,))
    except psycopg.errors.InvalidTextRepresentation as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid job_id: {exc}",
        )

    row = await cur.fetchone()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    db_status, path, department, error = row
    return JobProgressResponse(
        job_id=job_id,
        status=db_status,  # type: ignore[arg-type]
        path=path,
        department=department,
        error=error,
    )
