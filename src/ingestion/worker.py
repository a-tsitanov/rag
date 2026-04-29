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
from langchain_core.embeddings import Embeddings
from loguru import logger

from src.config import settings
from src.ingestion.chunker import Chunk, SemanticChunker
from src.ingestion.identifiers import (
    build_augment_block,
    build_custom_kg_payload,
    extract_identifiers,
)
from src.ingestion.parser import DocumentParser, ParsedDocument
from src.llm_client import LLMClient
from src.storage.sparse_encoder import SparseEncoder

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
    summary: str = ""


# ── type aliases ──────────────────────────────────────────────────────

LightRAGInserter = Callable[[str], Coroutine[Any, Any, None]]
LightRAGCustomKGInserter = Callable[[dict], Coroutine[Any, Any, None]]
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
        llm_client: LLMClient | None = None,
        parser: DocumentParser | None = None,
        chunker: SemanticChunker | None = None,
        embeddings: Embeddings | None = None,
        sparse_encoder: SparseEncoder | None = None,
        lightrag_custom_kg_inserter: LightRAGCustomKGInserter | None = None,
    ):
        self._parser = parser or DocumentParser()
        self._chunker = chunker
        self._vectorstore = vectorstore
        self._lightrag_insert = lightrag_inserter
        self._lightrag_custom_kg = lightrag_custom_kg_inserter
        self._pg_status = pg_status_updater
        self._llm = llm_client
        self._embeddings = embeddings
        self._sparse_encoder = sparse_encoder

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
            summary=payload.summary,
        )

    async def _write_hybrid(
        self,
        texts: list[str],
        ids: list[str],
        doc_id: str,
        department: str,
        doc_type: str,
        created_at: int,
    ) -> None:
        """Write dense + sparse vectors to Milvus via pymilvus."""
        from pymilvus import MilvusClient

        dense_vectors = await asyncio.to_thread(
            self._embeddings.embed_documents, texts,
        )
        sparse_vectors = [
            self._sparse_encoder.encode_document(t) for t in texts
        ]

        data = [
            {
                "id": cid,
                "content": text,
                "embedding": dvec,
                "sparse_embedding": svec,
                "doc_id": doc_id,
                "department": department,
                "doc_type": doc_type,
                "created_at": created_at,
            }
            for cid, text, dvec, svec in zip(
                ids, texts, dense_vectors, sparse_vectors,
            )
        ]

        uri = f"http://{settings.milvus.host}:{settings.milvus.port}"
        client = MilvusClient(uri=uri)
        try:
            await asyncio.to_thread(
                client.insert, settings.milvus.collection, data,
            )
        finally:
            client.close()

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

            # 1b. summarize (if enabled + LLM client available)
            summary = ""
            if settings.ingestion.summary_enabled and self._llm and doc.text:
                t1 = self._now()
                try:
                    resp = await self._llm.chat(
                        model=settings.effective_llm_model,
                        messages=[
                            {"role": "system", "content": settings.ingestion.summary_prompt},
                            {"role": "user", "content": doc.text[:8000]},
                        ],
                    )
                    summary = resp["message"]["content"].strip()
                    timings.append(StepTiming("summary", self._now() - t1))
                except Exception as exc:
                    logger.warning(
                        "summary generation failed  doc_id={doc_id} err={err}",
                        doc_id=doc_id, err=exc,
                    )

            # 2. chunk (langchain SemanticChunker)
            t0 = self._now()
            if self._chunker:
                chunks = await self._chunker.chunk(doc, doc_id=doc_id)
            else:
                raise ValueError("chunker not configured")
            timings.append(StepTiming("chunk", self._now() - t0))

            if not chunks:
                raise ValueError("chunker produced zero chunks")

            # 3. vectorstore write (embed + write → Milvus)
            t0 = self._now()
            now_ts = int(time.time())
            ids = [c.chunk_id for c in chunks]
            texts = [c.content for c in chunks]

            if self._sparse_encoder and self._embeddings:
                # Phase 3: hybrid write — dense + sparse via pymilvus
                await self._write_hybrid(
                    texts, ids, doc_id, department, doc_type, now_ts,
                )
            else:
                # Fallback: langchain write (dense only)
                lc_docs = [
                    LCDocument(
                        page_content=text,
                        metadata={
                            "doc_id": doc_id,
                            "department": department,
                            "doc_type": doc_type,
                            "created_at": now_ts,
                        },
                    )
                    for text in texts
                ]
                await asyncio.to_thread(
                    self._vectorstore.add_documents, lc_docs, ids=ids,
                )
            timings.append(StepTiming("vectorstore", self._now() - t0))

            # 4. LightRAG entity extraction (свой pipeline)
            #
            # Two-stage handoff (see retrieval-quality plan, Stage C):
            #   a. extract_identifiers — deterministic regex/lib pass
            #      yields canonical forms (E.164 phones, ISO dates, INN
            #      with valid checksum, ...).
            #   b. ainsert_custom_kg — guarantees a Neo4j node exists
            #      with the canonical name BEFORE the LLM extraction
            #      runs over the same text.  Subsequent LLM-extracted
            #      relationships then attach to the canonical node.
            #   c. text is augmented with a `Канонические идентификаторы`
            #      block so the LLM (cued by Stage A's system prompt)
            #      uses canonical forms when building relations.
            #
            # The custom-KG inserter is optional — when not wired
            # (older configs, tests with simpler fakes) we fall back to
            # plain ainsert(text) and graph-fragmentation persists.
            t0 = self._now()
            text_for_rag = doc.text
            if self._lightrag_custom_kg is not None:
                idents = extract_identifiers(doc.text)
                if idents:
                    payload = build_custom_kg_payload(
                        idents,
                        doc_id=doc_id,
                        file_path=str(path),
                        text=doc.text,
                    )
                    try:
                        await self._lightrag_custom_kg(payload)
                    except Exception as exc:
                        logger.warning(
                            "ainsert_custom_kg failed (continuing with "
                            "plain ainsert)  doc_id={doc_id} err={err}",
                            doc_id=doc_id, err=exc,
                        )
                    text_for_rag = doc.text + build_augment_block(idents)
                    logger.info(
                        "identifiers extracted  doc_id={doc_id} "
                        "count={n}",
                        doc_id=doc_id, n=len(idents),
                    )
            await self._lightrag_insert(text_for_rag)
            timings.append(StepTiming("lightrag", self._now() - t0))

            # 5. PG status → completed (+ summary if generated)
            t0 = self._now()
            payload = PGStatusPayload(
                doc_id=doc_id, status="completed",
                path=str(path), department=department, doc_type=doc_type,
            )
            payload.summary = summary
            await self._write_status(payload)
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
