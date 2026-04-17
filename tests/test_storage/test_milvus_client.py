"""Tests for Milvus helpers (health check + dataclass contracts)."""

import dataclasses

import pytest

from src.storage.milvus_client import Document, SearchResult


EXPECTED_MILVUS_FIELDS = {
    "id", "content", "embedding", "doc_id",
    "department", "created_at", "doc_type",
}


def test_document_dataclass_matches_expected_schema():
    doc_fields = {f.name for f in dataclasses.fields(Document)}
    assert doc_fields == EXPECTED_MILVUS_FIELDS


def test_search_result_dataclass_has_expected_fields():
    expected = {"id", "content", "doc_id", "department", "doc_type", "score"}
    sr_fields = {f.name for f in dataclasses.fields(SearchResult)}
    assert sr_fields == expected


# NOTE: integration-тест check_milvus_health намеренно НЕ включён в
# unit-набор — MilvusClient gRPC constructor зависает навсегда когда
# сервер недоступен, и asyncio.wait_for не может отменить нативный
# thread.  Проверяется через scripts/smoke.sh → /health endpoint.
