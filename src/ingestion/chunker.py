"""Semantic chunker: splits a ParsedDocument into overlapping Chunk objects.

Algorithm
---------
1. Split section contents into sentences (nltk ``sent_tokenize``).
2. Embed every sentence with the provided ``embed_fn`` (default: BGE-M3).
3. Greedily accumulate sentences into a chunk while:
   a. cosine similarity between the sentence and the chunk centroid ≥ threshold, AND
   b. total token count ≤ ``max_tokens``.
4. When either condition breaks, seal the chunk and start a new one.
5. Post-process: prepend the last ``overlap`` tokens of the previous chunk
   to the beginning of each subsequent chunk.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import tiktoken

from src.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)

# ── sentence tokenisation (nltk with regex fallback) ──────────────────

try:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
except Exception:  # ImportError / LookupError / OSError
    _nltk_sent_tokenize = None

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _sent_tokenize(text: str) -> list[str]:
    if _nltk_sent_tokenize is not None:
        return _nltk_sent_tokenize(text)
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


# ── data class ────────────────────────────────────────────────────────

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


# ── helpers ───────────────────────────────────────────────────────────


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def _default_embed_fn() -> EmbedFn:
    """Lazy-load BGE-M3 via *sentence-transformers*.

    Install with ``pip install sentence-transformers`` (~2 GB including
    torch).  If the package is missing an ``ImportError`` is raised at
    chunking time — not at import time — so the rest of the codebase is
    unaffected.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "SemanticChunker needs sentence-transformers for the default "
            "embed_fn.  Install it with:\n"
            "  pip install sentence-transformers\n"
            "Or pass a custom embed_fn to the constructor."
        )

    model = SentenceTransformer("BAAI/bge-m3")

    def _encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, normalize_embeddings=True)

    return _encode


# ── chunker ───────────────────────────────────────────────────────────


class SemanticChunker:
    """Split a :class:`ParsedDocument` into semantically coherent chunks."""

    def __init__(
        self,
        *,
        embed_fn: EmbedFn | None = None,
        max_tokens: int = 512,
        overlap: int = 50,
        similarity_threshold: float = 0.8,
    ):
        self._embed_fn = embed_fn or _default_embed_fn()
        self._max_tokens = max_tokens
        self._overlap = overlap
        self._sim_threshold = similarity_threshold
        self._enc = tiktoken.get_encoding("cl100k_base")

    # ── token helpers ─────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _tail_text(self, text: str, n_tokens: int) -> str:
        """Return the decoded text of the last *n_tokens* tokens."""
        toks = self._enc.encode(text)
        tail = toks[-n_tokens:] if len(toks) > n_tokens else toks
        return self._enc.decode(tail)

    # ── step 1: sentences ─────────────────────────────────────────────

    def _tagged_sentences(
        self, doc: ParsedDocument
    ) -> list[tuple[str, str]]:
        """Return ``(sentence, section_title)`` pairs."""
        pairs: list[tuple[str, str]] = []

        sources = doc.sections if doc.sections else []
        if not sources and doc.text:
            from src.ingestion.parser import Section

            sources = [Section(title="", content=doc.text, level=0)]

        for section in sources:
            text = section.content.strip()
            if not text:
                continue
            for sent in _sent_tokenize(text):
                sent = sent.strip()
                if sent:
                    pairs.append((sent, section.title))
        return pairs

    # ── step 3: greedy grouping ───────────────────────────────────────

    def _group(
        self,
        sentences: list[str],
        embeddings: np.ndarray,
        titles: list[str],
    ) -> list[tuple[list[str], str]]:
        """Group sentences by similarity + token budget.

        Returns a list of ``(sentences, section_title)`` tuples — one per
        future chunk (before overlap is applied).
        """
        groups: list[tuple[list[str], str]] = []

        cur: list[str] = [sentences[0]]
        cur_tokens = self._count_tokens(sentences[0])
        cur_title = titles[0]
        cur_centroid = embeddings[0].astype(np.float64)

        for i in range(1, len(sentences)):
            sent = sentences[i]
            emb = embeddings[i].astype(np.float64)
            sent_tokens = self._count_tokens(sent)

            sim = _cosine_sim(cur_centroid, emb)
            fits = cur_tokens + sent_tokens <= self._max_tokens

            if sim >= self._sim_threshold and fits:
                cur.append(sent)
                cur_tokens += sent_tokens
                n = len(cur)
                cur_centroid = cur_centroid * ((n - 1) / n) + emb / n
            else:
                groups.append((list(cur), cur_title))
                cur = [sent]
                cur_tokens = sent_tokens
                cur_title = titles[i]
                cur_centroid = emb.copy()

        if cur:
            groups.append((list(cur), cur_title))

        return groups

    # ── step 5: build chunks with overlap ─────────────────────────────

    def _build_chunks(
        self,
        groups: list[tuple[list[str], str]],
        doc_id: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        prev_raw = ""  # raw content of previous group (before overlap)

        for pos, (sents, section_title) in enumerate(groups):
            raw = " ".join(sents)

            if prev_raw and self._overlap > 0:
                overlap_text = self._tail_text(prev_raw, self._overlap)
                content = overlap_text + " " + raw
            else:
                content = raw

            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_{pos}",
                    doc_id=doc_id,
                    content=content,
                    position=pos,
                    section_title=section_title,
                    token_count=self._count_tokens(content),
                    metadata={},
                )
            )
            prev_raw = raw

        return chunks

    # ── public API ────────────────────────────────────────────────────

    async def chunk(self, doc: ParsedDocument, doc_id: str = "") -> list[Chunk]:
        if not doc_id:
            doc_id = str(uuid.uuid4())

        tagged = self._tagged_sentences(doc)
        if not tagged:
            return []

        sentences = [s for s, _ in tagged]
        titles = [t for _, t in tagged]

        # step 2: embed (supports sync or async embed_fn)
        vecs = self._embed_fn(sentences)
        if asyncio.iscoroutine(vecs):
            vecs = await vecs
        embeddings = np.asarray(vecs, dtype=np.float32)

        # step 3: group
        groups = self._group(sentences, embeddings, titles)

        # steps 4-5: overlap + build
        return self._build_chunks(groups, doc_id)
