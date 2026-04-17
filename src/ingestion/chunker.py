"""Semantic chunker (langchain-experimental).

Принимает ``OllamaEmbeddings`` напрямую — langchain ``SemanticChunker``
вызывает ``.embed_documents()`` синхронно внутри ``split_text()``.
Наш ``chunk()`` вызывает ``split_text`` через ``asyncio.to_thread``,
чтобы не блокировать event loop.

Post-process: overlap + section-title mapping + Chunk dataclass.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

import tiktoken
from langchain_core.embeddings import Embeddings
from loguru import logger

from src.config import settings
from src.ingestion.parser import ParsedDocument, Section


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    position: int
    section_title: str
    token_count: int
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    """Split a :class:`ParsedDocument` into semantically coherent chunks."""

    def __init__(
        self,
        *,
        embeddings: Embeddings,
        max_tokens: int = 512,
        overlap: int = 50,
        breakpoint_threshold_type: str | None = None,
        breakpoint_threshold_amount: float | None = None,
    ):
        from langchain_experimental.text_splitter import (
            SemanticChunker as LCSemanticChunker,
        )

        self._max_tokens = max_tokens
        self._overlap = overlap
        self._enc = tiktoken.get_encoding("cl100k_base")

        bp_type = breakpoint_threshold_type or settings.ingestion.breakpoint_type
        bp_amount = (
            breakpoint_threshold_amount
            if breakpoint_threshold_amount is not None
            else settings.ingestion.breakpoint_amount
        )

        kwargs: dict = {
            "embeddings": embeddings,
            "breakpoint_threshold_type": bp_type,
        }
        if bp_amount is not None:
            kwargs["breakpoint_threshold_amount"] = bp_amount

        self._splitter = LCSemanticChunker(**kwargs)

    # ── helpers ───────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _tail_text(self, text: str, n_tokens: int) -> str:
        toks = self._enc.encode(text)
        tail = toks[-n_tokens:] if len(toks) > n_tokens else toks
        return self._enc.decode(tail)

    @staticmethod
    def _sections_from_doc(doc: ParsedDocument) -> list[Section]:
        if doc.sections:
            return [s for s in doc.sections if s.content.strip()]
        if doc.text:
            return [Section(title="", content=doc.text, level=0)]
        return []

    @staticmethod
    def _find_section_title(chunk_text: str, sections: list[Section]) -> str:
        for section in sections:
            if chunk_text[:80] in section.content:
                return section.title
        return sections[0].title if sections else ""

    # ── public API ────────────────────────────────────────────────────

    async def chunk(
        self, doc: ParsedDocument, doc_id: str = "",
    ) -> list[Chunk]:
        if not doc_id:
            doc_id = str(uuid.uuid4())

        sections = self._sections_from_doc(doc)
        if not sections:
            return []

        full_text = "\n\n".join(s.content.strip() for s in sections)

        raw_chunks: list[str] = await asyncio.to_thread(
            self._splitter.split_text, full_text,
        )

        if not raw_chunks:
            return []

        chunks: list[Chunk] = []
        prev_raw = ""

        for pos, raw in enumerate(raw_chunks):
            if prev_raw and self._overlap > 0:
                overlap_text = self._tail_text(prev_raw, self._overlap)
                content = overlap_text + " " + raw
            else:
                content = raw

            section_title = self._find_section_title(raw, sections)

            chunks.append(Chunk(
                chunk_id=f"{doc_id}_{pos}",
                doc_id=doc_id,
                content=content,
                position=pos,
                section_title=section_title,
                token_count=self._count_tokens(content),
                metadata={},
            ))
            prev_raw = raw

        logger.debug(
            "chunked  doc_id={doc_id} chunks={n}",
            doc_id=doc_id, n=len(chunks),
        )
        return chunks
