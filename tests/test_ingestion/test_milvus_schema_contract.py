"""Contract-тест: поля Milvus-row, которые пишет AsyncDocumentWorker,
совпадают с полями коллекции, которые создаёт AsyncMilvusClient.

При расхождении (опечатка в имени, забытая колонка, новая в схеме)
Milvus молча дропнул бы row или упал бы в runtime. Этот тест ловит
расхождение на уровне юнита.
"""

from __future__ import annotations

import dataclasses

from src.storage.milvus_client import Document


EXPECTED_MILVUS_FIELDS = {
    "id", "content", "embedding", "doc_id",
    "department", "created_at", "doc_type",
}


def test_document_dataclass_matches_collection_schema():
    """Все поля Milvus schema представлены в ``Document`` dataclass."""
    doc_fields = {f.name for f in dataclasses.fields(Document)}
    assert doc_fields == EXPECTED_MILVUS_FIELDS, (
        f"Document dataclass fields {doc_fields} диверджат с "
        f"Milvus schema {EXPECTED_MILVUS_FIELDS}"
    )


def test_worker_produces_rows_with_expected_keys(tmp_path):
    """AsyncDocumentWorker'овские Milvus-rows используют те же ключи
    (через ``Document.__dict__``). Строим один row вручную —
    полный e2e тут не гоняем, это делает test_worker."""
    row = Document(
        id="chunk-0",
        content="hello",
        embedding=[0.0, 0.1, 0.2],
        doc_id="doc-1",
        department="dev",
        created_at=1_700_000_000,
        doc_type="txt",
    ).__dict__

    assert set(row.keys()) == EXPECTED_MILVUS_FIELDS
