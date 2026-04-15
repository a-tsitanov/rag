"""Tests for SemanticChunker.

Uses a lightweight deterministic embedding function so the test suite
does not need ``sentence-transformers`` or a GPU.
"""

from pathlib import Path

import numpy as np
import pytest

from src.ingestion.chunker import Chunk, SemanticChunker
from src.ingestion.parser import DocumentParser, ParsedDocument, Section

FIXTURES = Path(__file__).parent / "fixtures"

# ── test embedding function (no model download) ──────────────────────


def _deterministic_embed(texts: list[str]) -> np.ndarray:
    """Hash-seeded unit vectors — fast, deterministic, no external model."""
    vecs = []
    for t in texts:
        seed = hash(t) & 0xFFFF_FFFF
        rng = np.random.RandomState(seed)
        v = rng.randn(64).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-10
        vecs.append(v)
    return np.array(vecs, dtype=np.float32)


def _identical_embed(texts: list[str]) -> np.ndarray:
    """All texts get the *same* unit vector → similarity is always 1.0.

    Useful for testing pure token-budget splitting without semantic
    boundaries breaking chunks apart.
    """
    v = np.ones(64, dtype=np.float32)
    v /= np.linalg.norm(v)
    return np.tile(v, (len(texts), 1))


# ── fixtures ──────────────────────────────────────────────────────────

LONG_TEXT = (
    "The enterprise knowledge base system stores documents from multiple departments. "
    "It supports PDF, DOCX, PPTX, and plain text formats. "
    "Documents are parsed and split into chunks for vector search. "
    "Each chunk is embedded using a BGE-M3 model and stored in Milvus. "
    "The system also builds a knowledge graph in Neo4j. "
    "Entities and relations are extracted during ingestion. "
    "Users can search the knowledge base with natural language queries. "
    "The search pipeline retrieves relevant chunks via cosine similarity. "
    "Results are then re-ranked and used to generate answers with an LLM. "
    "The architecture uses Redis as a task queue for asynchronous processing. "
    "PostgreSQL stores document metadata and processing status. "
    "The entire stack runs in Docker Compose for easy deployment. "
    "Monitoring is done through structured logging and health checks. "
    "Access control is based on department-level permissions. "
    "The API is built with FastAPI and documented with OpenAPI. "
)


def _make_long_doc(n_repeats: int = 5) -> ParsedDocument:
    """Build a document long enough to produce multiple chunks."""
    sections = [
        Section(
            title=f"Section {i + 1}",
            content=LONG_TEXT,
            level=1,
        )
        for i in range(n_repeats)
    ]
    full_text = "\n\n".join(s.content for s in sections)
    return ParsedDocument(
        text=full_text,
        metadata={"doc_type": "synthetic"},
        sections=sections,
    )


# ── basic structure ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chunk_returns_list_of_chunks():
    doc = _make_long_doc()
    chunker = SemanticChunker(embed_fn=_deterministic_embed)
    chunks = await chunker.chunk(doc, doc_id="test-doc")

    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.doc_id == "test-doc"
        assert c.chunk_id.startswith("test-doc_")
        assert c.position >= 0
        assert c.token_count > 0
        assert len(c.content) > 0
        assert isinstance(c.metadata, dict)


# ── token budget ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_raw_group_tokens_within_budget():
    """Each chunk's *original* sentences (before overlap) ≤ max_tokens."""
    doc = _make_long_doc(n_repeats=8)
    chunker = SemanticChunker(
        embed_fn=_identical_embed,  # no semantic breaks → pure budget splits
        max_tokens=512,
        overlap=0,  # disable overlap so content = raw group
    )
    chunks = await chunker.chunk(doc, doc_id="budget")

    assert len(chunks) >= 2, "document should be long enough to split"
    for c in chunks:
        assert c.token_count <= 520, (
            f"chunk {c.position} has {c.token_count} tokens "
            f"(max_tokens=512, small rounding tolerance)"
        )


# ── overlap ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_overlap_between_consecutive_chunks():
    doc = _make_long_doc(n_repeats=8)
    chunker = SemanticChunker(
        embed_fn=_identical_embed,
        max_tokens=512,
        overlap=50,
    )
    chunks = await chunker.chunk(doc, doc_id="overlap")

    assert len(chunks) >= 2

    for i in range(1, len(chunks)):
        prev = chunks[i - 1].content
        curr = chunks[i].content

        # The overlap text (tail of prev) must appear at the start of curr
        # Find the overlap by checking shared content
        # We check that curr starts with text derived from prev's tail
        overlap_candidate = curr.split(" ")[0:10]  # first ~10 words
        joined = " ".join(overlap_candidate)
        assert joined in prev, (
            f"chunk {i} should start with overlap from chunk {i - 1}"
        )


# ── section titles ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_section_titles_preserved():
    doc = ParsedDocument(
        text="",
        metadata={},
        sections=[
            Section(title="Introduction", content="This is the introduction. It sets the scene.", level=1),
            Section(title="Methods", content="We used advanced techniques. The approach was novel.", level=1),
        ],
    )
    chunker = SemanticChunker(
        embed_fn=_deterministic_embed,
        max_tokens=512,
        overlap=0,
    )
    chunks = await chunker.chunk(doc, doc_id="sec")

    assert len(chunks) >= 1
    titles = {c.section_title for c in chunks}
    assert titles & {"Introduction", "Methods"}


# ── empty document ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_document_returns_no_chunks():
    doc = ParsedDocument(text="", metadata={}, sections=[])
    chunker = SemanticChunker(embed_fn=_deterministic_embed)
    assert await chunker.chunk(doc) == []


# ── real PDF fixture ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chunk_real_pdf():
    """Parse the PDF fixture, chunk it, and print the chunks."""
    parser = DocumentParser()
    doc = parser.parse(FIXTURES / "sample.pdf")

    chunker = SemanticChunker(embed_fn=_deterministic_embed, max_tokens=512, overlap=50)
    chunks = await chunker.chunk(doc, doc_id="pdf-sample")

    assert len(chunks) >= 1

    print("\n\n=== Chunks from sample.pdf ===")
    for c in chunks:
        print(
            f"\n--- chunk {c.position} | section: {c.section_title!r} "
            f"| tokens: {c.token_count} ---"
        )
        print(c.content[:200])

    # structural checks
    for c in chunks:
        assert c.doc_id == "pdf-sample"
        assert c.chunk_id == f"pdf-sample_{c.position}"
        assert c.token_count > 0
        assert c.section_title  # PDF sections are "Page 1", "Page 2"


# ── real DOCX fixture ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chunk_real_docx():
    parser = DocumentParser()
    doc = parser.parse(FIXTURES / "sample.docx")

    chunker = SemanticChunker(embed_fn=_deterministic_embed, max_tokens=512, overlap=50)
    chunks = await chunker.chunk(doc, doc_id="docx-sample")

    assert len(chunks) >= 1

    # verify heading titles flow through
    titles = {c.section_title for c in chunks}
    assert titles & {"Introduction", "Details", "Conclusion"}


# ── doc_id auto-generated when omitted ────────────────────────────────


@pytest.mark.asyncio
async def test_doc_id_auto_generated():
    doc = ParsedDocument(
        text="One sentence.",
        metadata={},
        sections=[Section(title="T", content="One sentence.", level=0)],
    )
    chunker = SemanticChunker(embed_fn=_deterministic_embed)
    chunks = await chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].doc_id  # non-empty UUID
    assert chunks[0].chunk_id.endswith("_0")
