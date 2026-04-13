from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def test_text_splitting():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    text = "Hello world. " * 200
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= settings.chunk_size + 50 for c in chunks)
