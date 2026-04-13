from pathlib import Path

import pytest

from src.ingestion.parser import DocumentParser, ParsedDocument, Section

FIXTURES = Path(__file__).parent / "fixtures"

parser = DocumentParser()


# ── helpers ───────────────────────────────────────────────────────────


def _assert_base(doc: ParsedDocument, *, min_words: int = 3):
    """Every parsed document must meet these invariants."""
    assert isinstance(doc, ParsedDocument)
    assert isinstance(doc.text, str)
    assert len(doc.text) > 0
    assert doc.metadata["word_count"] >= min_words
    assert "doc_type" in doc.metadata
    assert isinstance(doc.sections, list)
    assert len(doc.sections) > 0
    for sec in doc.sections:
        assert isinstance(sec, Section)


# ── PDF ───────────────────────────────────────────────────────────────


def test_parse_pdf():
    doc = parser.parse(FIXTURES / "sample.pdf")
    _assert_base(doc)
    assert doc.metadata["doc_type"] == "pdf"
    assert doc.metadata.get("pages") == 2
    assert doc.metadata.get("title") == "Test PDF Document"
    assert doc.metadata.get("author") == "Test Author"
    assert "Enterprise Knowledge Base" in doc.text
    assert len(doc.sections) == 2  # one per page


# ── DOCX ──────────────────────────────────────────────────────────────


def test_parse_docx():
    doc = parser.parse(FIXTURES / "sample.docx")
    _assert_base(doc)
    assert doc.metadata["doc_type"] == "docx"
    assert doc.metadata.get("title") == "Test DOCX Document"
    assert doc.metadata.get("author") == "Test Author"
    assert "introduction" in doc.text.lower()

    # heading structure preserved
    headings = [s.title for s in doc.sections if s.level >= 1]
    assert "Introduction" in headings
    assert "Details" in headings
    assert "Conclusion" in headings


# ── PPTX ──────────────────────────────────────────────────────────────


def test_parse_pptx():
    doc = parser.parse(FIXTURES / "sample.pptx")
    _assert_base(doc)
    assert doc.metadata["doc_type"] == "pptx"
    assert doc.metadata.get("pages") == 3
    assert doc.metadata.get("title") == "Test Presentation"
    assert doc.metadata.get("author") == "Test Author"
    assert len(doc.sections) == 3  # one per slide
    assert "Architecture" in doc.text


# ── TXT ───────────────────────────────────────────────────────────────


def test_parse_txt():
    doc = parser.parse(FIXTURES / "sample.txt")
    _assert_base(doc)
    assert doc.metadata["doc_type"] == "txt"
    assert "Enterprise Knowledge Base" in doc.text
    assert doc.metadata.get("encoding")


# ── Markdown ──────────────────────────────────────────────────────────


def test_parse_md():
    doc = parser.parse(FIXTURES / "sample.md")
    _assert_base(doc)
    assert doc.metadata["doc_type"] == "md"
    assert "markdown" in doc.text.lower()


# ── EML ───────────────────────────────────────────────────────────────


def test_parse_eml():
    doc = parser.parse(FIXTURES / "sample.eml")
    _assert_base(doc)
    assert doc.metadata["doc_type"] == "eml"
    assert doc.metadata.get("title") == "Enterprise KB Update"
    assert doc.metadata.get("author") == "sender@example.com"
    assert "Knowledge Base" in doc.text

    # sections: Subject, Body
    section_titles = [s.title for s in doc.sections]
    assert "Subject" in section_titles
    assert "Body" in section_titles


# ── error handling ────────────────────────────────────────────────────


def test_parse_nonexistent_file_returns_partial():
    doc = parser.parse(Path("/tmp/does_not_exist.pdf"))
    assert isinstance(doc, ParsedDocument)
    assert "error" in doc.metadata


def test_parse_corrupt_file_returns_partial(tmp_path):
    bad = tmp_path / "corrupt.docx"
    bad.write_bytes(b"not a real docx file")
    doc = parser.parse(bad)
    assert isinstance(doc, ParsedDocument)
    assert "error" in doc.metadata


# ── unsupported extension falls back to text ─────────────────────────


def test_parse_unknown_extension_as_text(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("col1,col2\nval1,val2\n")
    doc = parser.parse(f)
    assert isinstance(doc, ParsedDocument)
    assert "col1" in doc.text
