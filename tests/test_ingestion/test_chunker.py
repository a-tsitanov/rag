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

# Разнородные тематические секции — langchain SemanticChunker строит
# breakpoint'ы по межсентенсиальной семантической дистанции, поэтому
# нужна реальная тематическая вариативность, а не повтор одного блока.
_SECTION_TEXTS = [
    (
        "Architecture",
        "The enterprise knowledge base stores documents from multiple departments. "
        "It supports PDF, DOCX, PPTX, and plain text formats. "
        "Documents are parsed and split into chunks for vector search. "
        "Each chunk is embedded using a sentence-transformer model. "
        "The system also builds a knowledge graph in Neo4j for entity linking.",
    ),
    (
        "Ingestion",
        "Uploaded files land on a shared volume between API and worker. "
        "A RabbitMQ message triggers the asynchronous ingestion task. "
        "The worker parses the file, runs semantic chunking, and computes embeddings. "
        "Chunks are written to Milvus with cosine-based HNSW indexing. "
        "LightRAG extracts entities and relations for the knowledge graph.",
    ),
    (
        "Search",
        "Users query the knowledge base with natural language. "
        "The query is embedded and matched against Milvus candidates. "
        "A cross-encoder reranker refines the top results. "
        "LightRAG generates a coherent answer from the graph context. "
        "The response includes source citations with chunk IDs and scores.",
    ),
    (
        "Infrastructure",
        "PostgreSQL stores document metadata and processing status. "
        "Redis was replaced with RabbitMQ for durable task queuing. "
        "Docker Compose orchestrates all services including Ollama for LLM inference. "
        "The stack includes Milvus with etcd and MinIO for object storage. "
        "Monitoring relies on structured logging via loguru.",
    ),
    (
        "Security",
        "Access control is department-based via Milvus partition keys. "
        "API authentication requires an X-API-Key header on every request. "
        "Keys are managed through environment variables and rotated per deployment. "
        "CORS policies are enforced at the middleware level. "
        "No PII is stored in logs thanks to loguru's diagnose=False setting.",
    ),
    (
        "Testing",
        "Unit tests cover the chunker, parser, worker, and storage clients. "
        "Integration tests use testcontainers for Neo4j and milvus-lite for Milvus. "
        "The smoke script performs end-to-end verification with curl and jq. "
        "Contract tests ensure Milvus schema and worker row types stay in sync. "
        "All fixtures use deterministic embedding functions to avoid model downloads.",
    ),
]


def _make_long_doc(n_repeats: int = 5) -> ParsedDocument:
    """Build a document from diverse sections, optionally duplicating the
    set to reach the desired length.  Each section has unique content so
    langchain's semantic breakpoint detector has real variance to work with.
    """
    pool = _SECTION_TEXTS * max(1, (n_repeats + len(_SECTION_TEXTS) - 1) // len(_SECTION_TEXTS))
    selected = pool[:n_repeats]
    sections = [
        Section(title=title, content=content, level=1)
        for title, content in selected
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


# ── semantic splitting (langchain breakpoint) ─────────────────────────


@pytest.mark.asyncio
async def test_semantic_split_produces_multiple_chunks():
    """With varied embeddings langchain's breakpoint strategy produces
    at least 2 chunks on a long enough document."""
    doc = _make_long_doc(n_repeats=8)
    chunker = SemanticChunker(
        embed_fn=_deterministic_embed,
        max_tokens=512,
        overlap=0,
    )
    chunks = await chunker.chunk(doc, doc_id="breakpoint")

    assert len(chunks) >= 2, (
        "document should be long enough for at least one breakpoint"
    )
    for c in chunks:
        assert c.token_count > 0
        assert len(c.content) > 0


# ── overlap ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_overlap_between_consecutive_chunks():
    doc = _make_long_doc(n_repeats=8)
    chunker = SemanticChunker(
        embed_fn=_deterministic_embed,
        max_tokens=512,
        overlap=50,
    )
    chunks = await chunker.chunk(doc, doc_id="overlap")

    assert len(chunks) >= 2, "need ≥2 chunks for overlap test"

    for i in range(1, len(chunks)):
        prev = chunks[i - 1].content
        curr = chunks[i].content

        overlap_candidate = curr.split(" ")[0:10]
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
