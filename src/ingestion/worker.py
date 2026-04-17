"""Async document ingestion worker.

``AsyncDocumentWorker`` orchestrates the full pipeline for a single file:
parse → chunk → vectorstore.add_documents (embed+write) →
LightRAG entity extraction → PG status.

``BatchProcessor`` fans out many files concurrently behind an
``asyncio.Semaphore``.

Dependencies are injected through the constructor — the entire pipeline
is testable without live services.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

from langchain_core.documents import Document as LCDocument
from loguru import logger

from src.ingestion.chunker import Chunk, SemanticChunker
from src.ingestion.parser import DocumentParser, ParsedDocument

# ── result / status ───────────────────────────────────────────────────


@dataclass
class StepTiming:
    name: str
    elapsed_s: float


@dataclass
class ProcessResult:
    doc_id: str
    path: str
    status: str  # "completed" | "failed"
    chunks: int = 0
    error: str = ""
    timings: list[StepTiming] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return sum(t.elapsed_s for t in self.timings)


@dataclass
class PGStatusPayload:
    doc_id: str
    status: str
    path: str = ""
    department: str = ""
    doc_type: str = ""
    error: str = ""


# ── type aliases ──────────────────────────────────────────────────────

LightRAGInserter = Callable[[str], Coroutine[Any, Any, None]]
PGStatusUpdater = Callable[..., Coroutine[Any, Any, None]]

# Vectorstore protocol — любой объект с add_documents (langchain Milvus,
# InMemoryVectorStore, fake в тестах).
VectorStoreProto = Any


# ── helpers ───────────────────────────────────────────────────────────


def _doc_type_from_path(path: Path) -> str:
    return path.suffix.lstrip(".").lower()


# ── worker ────────────────────────────────────────────────────────────


class AsyncDocumentWorker:
    """Process a single document through the full ingestion pipeline."""

    def __init__(
        self,
        *,
        vectorstore: VectorStoreProto,
        lightrag_inserter: LightRAGInserter,
        pg_status_updater: PGStatusUpdater,
        parser: DocumentParser | None = None,
        chunker: SemanticChunker | None = None,
    ):
        self._parser = parser or DocumentParser()
        self._chunker = chunker
        self._vectorstore = vectorstore
        self._lightrag_insert = lightrag_inserter
        self._pg_status = pg_status_updater

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    async def _write_status(self, payload: PGStatusPayload) -> None:
        await self._pg_status(
            doc_id=payload.doc_id,
            status=payload.status,
            path=payload.path,
            department=payload.department,
            doc_type=payload.doc_type,
            error=payload.error,
        )

    async def process_document(
        self,
        path: Path,
        *,
        doc_id: str | None = None,
        department: str = "",
    ) -> ProcessResult:
        doc_id = doc_id or str(uuid.uuid4())
        path = Path(path)
        doc_type = _doc_type_from_path(path)
        timings: list[StepTiming] = []
        chunks: list[Chunk] = []

        try:
            # 0. mark processing
            await self._write_status(PGStatusPayload(
                doc_id=doc_id, status="processing",
                path=str(path), department=department, doc_type=doc_type,
            ))

            # 1. parse
            t0 = self._now()
            doc: ParsedDocument = self._parser.parse(path)
            timings.append(StepTiming("parse", self._now() - t0))

            if not doc.text:
                raise ValueError(f"parser returned empty text for {path}")

            department = department or doc.metadata.get("department", "")

            # 2. chunk (langchain SemanticChunker)
            t0 = self._now()
            if self._chunker:
                chunks = await self._chunker.chunk(doc, doc_id=doc_id)
            else:
                raise ValueError("chunker not configured")
            timings.append(StepTiming("chunk", self._now() - t0))

            if not chunks:
                raise ValueError("chunker produced zero chunks")

            # 3. vectorstore.add_documents (embed + write → Milvus)
            t0 = self._now()
            now_ts = int(time.time())
            lc_docs = [
                LCDocument(
                    page_content=c.content,
                    metadata={
                        "doc_id": doc_id,
                        "department": department,
                        "doc_type": doc_type,
                        "created_at": now_ts,
                    },
                )
                for c in chunks
            ]
            ids = [c.chunk_id for c in chunks]

            await asyncio.to_thread(
                self._vectorstore.add_documents, lc_docs, ids=ids,
            )
            timings.append(StepTiming("vectorstore", self._now() - t0))

            # 4. LightRAG entity extraction (свой pipeline)
            t0 = self._now()
            await self._lightrag_insert(doc.text)
            timings.append(StepTiming("lightrag", self._now() - t0))

            # 5. PG status → completed
            t0 = self._now()
            await self._write_status(PGStatusPayload(
                doc_id=doc_id, status="completed",
                path=str(path), department=department, doc_type=doc_type,
            ))
            timings.append(StepTiming("pg_status", self._now() - t0))

        except Exception as exc:
            logger.error(
                "process_document failed  doc_id={doc_id} path={path} error={err}",
                doc_id=doc_id, path=str(path), err=exc,
            )
            try:
                await self._write_status(PGStatusPayload(
                    doc_id=doc_id, status="failed",
                    path=str(path), department=department,
                    doc_type=doc_type, error=str(exc),
                ))
            except Exception as pg_exc:
                logger.warning(
                    "PG status update failed  doc_id={doc_id} err={err}",
                    doc_id=doc_id, err=pg_exc,
                )

            return ProcessResult(
                doc_id=doc_id, path=str(path), status="failed",
                chunks=len(chunks), error=str(exc), timings=timings,
            )

        self._log_timings(path, doc_id, chunks, timings)
        return ProcessResult(
            doc_id=doc_id, path=str(path), status="completed",
            chunks=len(chunks), timings=timings,
        )

    @staticmethod
    def _log_timings(
        path: Path, doc_id: str,
        chunks: list[Chunk], timings: list[StepTiming],
    ):
        parts = " | ".join(f"{t.name}={t.elapsed_s:.3f}s" for t in timings)
        total = sum(t.elapsed_s for t in timings)
        logger.info(
            "processed  file={file} doc_id={doc_id} chunks={n} "
            "total_s={total:.3f} [{parts}]",
            file=path.name, doc_id=doc_id, n=len(chunks),
            total=total, parts=parts,
        )


# ── batch processor ───────────────────────────────────────────────────


class BatchProcessor:
    def __init__(self, worker: AsyncDocumentWorker, concurrency: int = 10):
        self._worker = worker
        self._semaphore = asyncio.Semaphore(concurrency)

    async def _process_one(self, path: Path) -> ProcessResult:
        async with self._semaphore:
            return await self._worker.process_document(path)

    async def process_batch(
        self, paths: list[Path], concurrency: int = 10,
    ) -> list[ProcessResult]:
        self._semaphore = asyncio.Semaphore(concurrency)
        tasks = [self._process_one(p) for p in paths]
        return await asyncio.gather(*tasks)
