"""Tests for SemanticChunker (langchain-experimental backend).

Uses a lightweight deterministic embedding function so the test suite
does not need Ollama or a GPU.
"""

from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from src.ingestion.chunker import Chunk, SemanticChunker
from src.ingestion.parser import DocumentParser, ParsedDocument, Section

FIXTURES = Path(__file__).parent / "fixtures"


# ── fake embeddings ──────────────────────────────────────────────────


class FakeEmbeddings(Embeddings):
    """Hash-seeded unit vectors — fast, deterministic, no model needed."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) & 0xFFFF_FFFF)
            v = rng.randn(64).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-10
            vecs.append(v.tolist())
        return vecs

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ── fixtures ──────────────────────────────────────────────────────────


_SECTION_TEXTS = [
    ("Architecture",
     "The enterprise knowledge base stores documents from multiple departments. "
     "It supports PDF, DOCX, PPTX, and plain text formats. "
     "Documents are parsed and split into chunks for vector search. "
     "Each chunk is embedded using a sentence-transformer model. "
     "The system also builds a knowledge graph in Neo4j for entity linking."),
    ("Ingestion",
     "Uploaded files land on a shared volume between API and worker. "
     "A RabbitMQ message triggers the asynchronous ingestion task. "
     "The worker parses the file, runs semantic chunking, and computes embeddings. "
     "Chunks are written to Milvus with cosine-based HNSW indexing. "
     "LightRAG extracts entities and relations for the knowledge graph."),
    ("Search",
     "Users query the knowledge base with natural language. "
     "The query is embedded and matched against Milvus candidates. "
     "A cross-encoder reranker refines the top results. "
     "LightRAG generates a coherent answer from the graph context. "
     "The response includes source citations with chunk IDs and scores."),
    ("Infrastructure",
     "PostgreSQL stores document metadata and processing status. "
     "RabbitMQ provides durable task queuing for ingestion. "
     "Docker Compose orchestrates all services including Ollama for LLM inference. "
     "The stack includes Milvus with etcd and MinIO for object storage. "
     "Monitoring relies on structured logging via loguru."),
    ("Security",
     "Access control is department-based via Milvus partition keys. "
     "API authentication requires an X-API-Key header on every request. "
     "Keys are managed through environment variables and rotated per deployment. "
     "CORS policies are enforced at the middleware level. "
     "No PII is stored in logs thanks to loguru diagnose-off setting."),
    ("Testing",
     "Unit tests cover the chunker, parser, worker, and storage clients. "
     "Integration tests use testcontainers for Neo4j and milvus-lite for Milvus. "
     "The smoke script performs end-to-end verification with curl and jq. "
     "Contract tests ensure Milvus schema and worker row types stay in sync. "
     "All fixtures use deterministic embedding functions to avoid model downloads."),
]


def _make_long_doc(n_repeats: int = 5) -> ParsedDocument:
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
    chunker = SemanticChunker(embeddings=FakeEmbeddings())
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


# ── semantic splitting ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_semantic_split_produces_multiple_chunks():
    doc = _make_long_doc(n_repeats=8)
    chunker = SemanticChunker(embeddings=FakeEmbeddings(), overlap=0)
    chunks = await chunker.chunk(doc, doc_id="breakpoint")

    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count > 0


# ── overlap ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_overlap_between_consecutive_chunks():
    doc = _make_long_doc(n_repeats=8)
    chunker = SemanticChunker(embeddings=FakeEmbeddings(), overlap=50)
    chunks = await chunker.chunk(doc, doc_id="overlap")

    assert len(chunks) >= 2

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
    chunker = SemanticChunker(embeddings=FakeEmbeddings(), overlap=0)
    chunks = await chunker.chunk(doc, doc_id="sec")

    assert len(chunks) >= 1
    titles = {c.section_title for c in chunks}
    assert titles & {"Introduction", "Methods"}


# ── empty document ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_document_returns_no_chunks():
    doc = ParsedDocument(text="", metadata={}, sections=[])
    chunker = SemanticChunker(embeddings=FakeEmbeddings())
    assert await chunker.chunk(doc) == []


# ── real PDF fixture ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chunk_real_pdf():
    parser = DocumentParser()
    doc = parser.parse(FIXTURES / "sample.pdf")

    chunker = SemanticChunker(embeddings=FakeEmbeddings(), overlap=50)
    chunks = await chunker.chunk(doc, doc_id="pdf-sample")

    assert len(chunks) >= 1
    for c in chunks:
        assert c.doc_id == "pdf-sample"
        assert c.token_count > 0


# ── real DOCX fixture ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chunk_real_docx():
    parser = DocumentParser()
    doc = parser.parse(FIXTURES / "sample.docx")

    chunker = SemanticChunker(embeddings=FakeEmbeddings(), overlap=50)
    chunks = await chunker.chunk(doc, doc_id="docx-sample")

    assert len(chunks) >= 1


# ── doc_id auto-generated when omitted ────────────────────────────────


@pytest.mark.asyncio
async def test_doc_id_auto_generated():
    doc = ParsedDocument(
        text="One sentence.",
        metadata={},
        sections=[Section(title="T", content="One sentence.", level=0)],
    )
    chunker = SemanticChunker(embeddings=FakeEmbeddings())
    chunks = await chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].doc_id
    assert chunks[0].chunk_id.endswith("_0")
