from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str

    # ── retrieval mode ────────────────────────────────────────────────
    mode: str = Field(
        "hybrid",
        description="naive | local | global | hybrid | mix | bypass",
    )
    department: str | None = None
    top_k: int = 10
    user_id: str | None = None

    # ── metadata filters (Phase 1a) ──────────────────────────────────
    doc_type_filter: str | None = Field(
        None, description="Фильтр по расширению: 'pdf', 'docx', etc.",
    )
    created_after: int | None = Field(
        None, description="Unix timestamp — только документы после этой даты.",
    )
    created_before: int | None = Field(
        None, description="Unix timestamp — только документы до этой даты.",
    )

    # ── LightRAG response tuning ─────────────────────────────────────
    response_type: str = Field(
        "Multiple Paragraphs",
        description=(
            "Формат ответа LightRAG: 'Multiple Paragraphs', "
            "'Single Paragraph', 'List', 'JSON', 'Table'."
        ),
    )
    include_references: bool = Field(
        False,
        description="Добавить ссылки на source-чанки в конец ответа.",
    )

    # ── LightRAG QueryParam knobs (Phase 1b) ─────────────────────────
    chunk_top_k: int = Field(20, description="Сколько чанков из vector store для RAG-контекста.")
    max_entity_tokens: int = Field(6000, description="Бюджет токенов на описания entities.")
    max_relation_tokens: int = Field(8000, description="Бюджет на relations.")
    max_total_tokens: int = Field(30000, description="Общий бюджет контекста для LLM.")

    # ── smart features (Phase 2+4, пока заглушки) ────────────────────
    decompose: bool = Field(False, description="Декомпозировать запрос на подвопросы (Phase 2b).")
    agentic: bool = Field(False, description="Итеративный multi-hop поиск (Phase 4).")


class SourceCitation(BaseModel):
    doc_id: str
    chunk_id: str
    position: int
    content: str
    score: float
    department: str = ""
    doc_type: str = ""


class SearchResponse(BaseModel):
    query: str
    answer: str
    mode: str
    sources: list[SourceCitation] = Field(default_factory=list)
    latency_ms: float = 0.0
