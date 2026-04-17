"""Semantic chunker built on top of ``langchain_experimental.text_splitter.SemanticChunker``.

Algorithm
---------
1. Concatenate all section content of the ``ParsedDocument`` into a single
   text (sections are joined with a double newline separator so the
   original paragraph structure is preserved for the splitter).
2. Hand the text to langchain's ``SemanticChunker`` which:
   a. Splits into sentences via regex.
   b. Embeds every sentence.
   c. Computes cosine distances between neighbouring sentence groups.
   d. Introduces breakpoints according to the configured strategy
      (``percentile`` | ``standard_deviation`` | ``interquartile`` |
      ``gradient``).
3. Post-process the resulting chunk strings:
   a. Count tokens (tiktoken ``cl100k_base``).
   b. Apply overlap: prepend the last ``overlap`` tokens of the previous
      chunk to the beginning of the next.
   c. Map each chunk back to its originating section title by substring
      containment.
   d. Build ``Chunk`` dataclass objects with ``doc_id``, ``position``, etc.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import tiktoken
from langchain_core.embeddings import Embeddings
from loguru import logger

from src.config import settings
from src.ingestion.parser import ParsedDocument, Section

# ── public types ──────────────────────────────────────────────────────

EmbedFn = Callable[[list[str]], np.ndarray]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    position: int
    section_title: str
    token_count: int
    metadata: dict = field(default_factory=dict)


# ── adapter: our async EmbedFn → langchain sync Embeddings ───────────


class _EmbeddingsAdapter(Embeddings):
    """Обёртка ``EmbedFn`` → ``langchain_core.embeddings.Embeddings``.

    ``SemanticChunker.split_text`` вызывает ``embed_documents`` синхронно.
    Мы запускаем ``split_text`` из async-контекста через
    ``asyncio.to_thread`` — внутри этого thread'а нет текущего event
    loop'а, поэтому ``asyncio.run`` безопасен.
    """

    def __init__(self, fn: EmbedFn) -> None:
        self._fn = fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self._fn(texts)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        arr = np.asarray(result, dtype=np.float32)
        return arr.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ── chunker ───────────────────────────────────────────────────────────


class SemanticChunker:
    """Split a :class:`ParsedDocument` into semantically coherent chunks
    using ``langchain_experimental.text_splitter.SemanticChunker``."""

    def __init__(
        self,
        *,
        embed_fn: EmbedFn,
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
        bp_amount = breakpoint_threshold_amount if breakpoint_threshold_amount is not None else settings.ingestion.breakpoint_amount

        kwargs: dict = {
            "embeddings": _EmbeddingsAdapter(embed_fn),
            "breakpoint_threshold_type": bp_type,
        }
        if bp_amount is not None:
            kwargs["breakpoint_threshold_amount"] = bp_amount

        self._splitter = LCSemanticChunker(**kwargs)

    # ── token helpers ─────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _tail_text(self, text: str, n_tokens: int) -> str:
        toks = self._enc.encode(text)
        tail = toks[-n_tokens:] if len(toks) > n_tokens else toks
        return self._enc.decode(tail)

    # ── section mapping ──────────────────────────────────────────────

    @staticmethod
    def _sections_from_doc(doc: ParsedDocument) -> list[Section]:
        if doc.sections:
            return [s for s in doc.sections if s.content.strip()]
        if doc.text:
            return [Section(title="", content=doc.text, level=0)]
        return []

    @staticmethod
    def _find_section_title(
        chunk_text: str, sections: list[Section],
    ) -> str:
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

        # langchain's split_text вызывает embed_documents синхронно →
        # выносим целиком в thread, чтобы не блокировать event-loop
        raw_chunks: list[str] = await asyncio.to_thread(
            self._splitter.split_text, full_text,
        )

        if not raw_chunks:
            return []

        # ── post-process: overlap + Chunk objects ────────────────
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
            "chunked  doc_id={doc_id}  raw_chunks={n_raw}  "
            "final_chunks={n_final}",
            doc_id=doc_id, n_raw=len(raw_chunks), n_final=len(chunks),
        )
        return chunks
