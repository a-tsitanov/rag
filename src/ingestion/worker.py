"""Async document ingestion worker.

``AsyncDocumentWorker`` orchestrates the full pipeline for a single file:
parse → chunk → embed → LightRAG entity extraction → Milvus write → PG status.

``BatchProcessor`` fans out many files concurrently behind an
``asyncio.Semaphore``.

Every external call is hidden behind *protocol-style* callables that are
injected through the constructor so the whole pipeline is testable without
live services.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

import numpy as np
from loguru import logger

from src.ingestion.chunker import Chunk, EmbedFn, SemanticChunker
from src.ingestion.parser import DocumentParser, ParsedDocument
from src.storage.milvus_client import Document as MilvusRow

# ── status / timing / result ──────────────────────────────────────────


@dataclass
class StepTiming:
    name: str
    elapsed_s: float


@dataclass
class ProcessResult:
    doc_id: str
    path: str
    status: str  # "completed" | "failed" | "skipped"
    chunks: int = 0
    error: str = ""
    timings: list[StepTiming] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return sum(t.elapsed_s for t in self.timings)


# ── typed pg-status payload (D7) ──────────────────────────────────────


@dataclass
class PGStatusPayload:
    """Аргументы записи в ``documents`` таблицу.

    Оформлено dataclass'ом, чтобы контракт между `AsyncDocumentWorker`,
    DI-провайдером `pg_status_updater` и тестами был явным и
    типизированным.  Добавление нового поля ломает один signature, а не
    молча дропается в ``**kwargs``.
    """

    doc_id: str
    status: str  # pending | processing | completed | failed
    path: str = ""
    department: str = ""
    doc_type: str = ""
    error: str = ""


# ── type aliases for injectable backends ──────────────────────────────

MilvusWriter = Callable[[list[dict]], Coroutine[Any, Any, None]]
LightRAGInserter = Callable[[str], Coroutine[Any, Any, None]]
PGStatusUpdater = Callable[..., Coroutine[Any, Any, None]]


# ── helpers ───────────────────────────────────────────────────────────


def _doc_type_from_path(path: Path) -> str:
    """Return file extension without dot ("pdf", "docx", …).

    Пустой suffix (файл без расширения) → ``""`` — колонка допускает.
    """
    return path.suffix.lstrip(".").lower()


# ── worker ────────────────────────────────────────────────────────────


class AsyncDocumentWorker:
    """Process a single document through the full ingestion pipeline."""

    def __init__(
        self,
        *,
        embed_fn: EmbedFn,
        milvus_writer: MilvusWriter,
        lightrag_inserter: LightRAGInserter,
        pg_status_updater: PGStatusUpdater,
        parser: DocumentParser | None = None,
        chunker: SemanticChunker | None = None,
        embedding_batch_size: int = 32,
        embedding_dim: int = 1024,
    ):
        self._parser = parser or DocumentParser()
        self._chunker = chunker or SemanticChunker(
            embed_fn=embed_fn, max_tokens=512, overlap=50,
        )
        self._embed_fn = embed_fn
        self._milvus_writer = milvus_writer
        self._lightrag_insert = lightrag_inserter
        self._pg_status = pg_status_updater
        self._batch_size = embedding_batch_size
        self._dim = embedding_dim

    # ── timing helper ─────────────────────────────────────────────────

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    async def _write_status(self, payload: PGStatusPayload) -> None:
        """Transport-layer конверсия: dataclass → kwargs для Postgres
        UPSERT.  Делаем здесь, чтобы наружу торчал только типизированный
        контракт `PGStatusPayload`."""
        await self._pg_status(
            doc_id=payload.doc_id,
            status=payload.status,
            path=payload.path,
            department=payload.department,
            doc_type=payload.doc_type,
            error=payload.error,
        )

    # ── public API ────────────────────────────────────────────────────

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
            # ── 0. mark processing ────────────────────────────────
            await self._write_status(PGStatusPayload(
                doc_id=doc_id, status="processing",
                path=str(path), department=department,
                doc_type=doc_type,
            ))

            # ── 1. parse ──────────────────────────────────────────
            t0 = self._now()
            doc: ParsedDocument = self._parser.parse(path)
            timings.append(StepTiming("parse", self._now() - t0))

            if not doc.text:
                raise ValueError(f"parser returned empty text for {path}")

            # caller's department wins; fall back to parser-derived value
            department = department or doc.metadata.get("department", "")

            # ── 2. chunk ──────────────────────────────────────────
            t0 = self._now()
            chunks = await self._chunker.chunk(doc, doc_id=doc_id)
            timings.append(StepTiming("chunk", self._now() - t0))

            if not chunks:
                raise ValueError("chunker produced zero chunks")

            # ── 3. batch embed ────────────────────────────────────
            t0 = self._now()
            texts = [c.content for c in chunks]
            all_embeddings: list[np.ndarray] = []

            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                vecs = self._embed_fn(batch)
                if asyncio.iscoroutine(vecs):
                    vecs = await vecs
                all_embeddings.append(np.asarray(vecs, dtype=np.float32))

            embeddings = np.concatenate(all_embeddings, axis=0)
            timings.append(StepTiming("embed", self._now() - t0))

            # ── 4. LightRAG entity extraction ─────────────────────
            t0 = self._now()
            await self._lightrag_insert(doc.text)
            timings.append(StepTiming("lightrag", self._now() - t0))

            # ── 5. Milvus batch write ─────────────────────────────
            t0 = self._now()
            now_ts = int(time.time())
            milvus_rows = [
                MilvusRow(
                    id=chunk.chunk_id,
                    content=chunk.content,
                    embedding=embeddings[idx].tolist(),
                    doc_id=doc_id,
                    department=department,
                    created_at=now_ts,
                    doc_type=doc_type,
                ).__dict__
                for idx, chunk in enumerate(chunks)
            ]
            await self._milvus_writer(milvus_rows)
            timings.append(StepTiming("milvus", self._now() - t0))

            # ── 6. PG status update ───────────────────────────────
            t0 = self._now()
            await self._write_status(PGStatusPayload(
                doc_id=doc_id, status="completed",
                path=str(path), department=department,
                doc_type=doc_type,
            ))
            timings.append(StepTiming("pg_status", self._now() - t0))

        except Exception as exc:
            logger.error(
                "process_document failed  doc_id={doc_id} path={path}  error={err}",
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
                    "could not update PG status  doc_id={doc_id}  pg_error={err}",
                    doc_id=doc_id, err=pg_exc,
                )

            return ProcessResult(
                doc_id=doc_id,
                path=str(path),
                status="failed",
                chunks=len(chunks),
                error=str(exc),
                timings=timings,
            )

        self._log_timings(path, doc_id, chunks, timings)

        return ProcessResult(
            doc_id=doc_id,
            path=str(path),
            status="completed",
            chunks=len(chunks),
            timings=timings,
        )

    # ── logging ───────────────────────────────────────────────────────

    @staticmethod
    def _log_timings(
        path: Path,
        doc_id: str,
        chunks: list[Chunk],
        timings: list[StepTiming],
    ):
        parts = " | ".join(f"{t.name}={t.elapsed_s:.3f}s" for t in timings)
        total = sum(t.elapsed_s for t in timings)
        logger.info(
            "processed  file={file} doc_id={doc_id} chunks={chunks} "
            "total_s={total:.3f} steps=[{parts}]",
            file=path.name, doc_id=doc_id, chunks=len(chunks),
            total=total, parts=parts,
        )


# ── batch processor ───────────────────────────────────────────────────


class BatchProcessor:
    """Process many documents concurrently with a semaphore limit."""

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
