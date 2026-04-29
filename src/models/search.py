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

    # ── smart features (Phase 2b + Phase 4) ────────────────────────────
    decompose: bool = Field(False, description="Декомпозировать запрос на подвопросы (Phase 2b).")
    agentic: bool = Field(False, description="Итеративный multi-hop поиск (Phase 4).")
    agentic_max_rounds: int = Field(
        3, ge=1, le=5,
        description="Макс. итераций agentic search (1-5).",
    )


class SourceCitation(BaseModel):
    doc_id: str
    chunk_id: str
    position: int
    content: str
    score: float
    department: str = ""
    doc_type: str = ""


class AgenticRoundStat(BaseModel):
    """Per-round telemetry from ``agentic_search`` (Phase 4).

    Lets UI / диагностика увидеть «что принёс каждый раунд» без чтения
    серверных логов — проще настраивать ``agentic_max_rounds`` и
    оценивать пользу follow-up'ов.
    """

    round: int = Field(..., description="1-based round number.")
    query: str = Field(
        ..., description="Запрос, использованный в этом раунде."
    )
    new_sources: int = Field(
        0, description="Сколько НОВЫХ чанков добавил раунд (после дедупа)."
    )
    new_entities: int = Field(
        0, description="Сколько НОВЫХ KG-сущностей добавил раунд."
    )
    new_relations: int = Field(
        0, description="Сколько НОВЫХ KG-отношений добавил раунд."
    )
    sufficient: bool | None = Field(
        None,
        description=(
            "Вердикт LLM-судьи: достаточно ли контекста. "
            "null когда судья не вызывался (early-exit на 'no new info')."
        ),
    )
    judge_reason: str = Field(
        "",
        description=(
            "Причина от судьи или 'no new info' для skipped-судьи "
            "(Stage G)."
        ),
    )


class SearchResponse(BaseModel):
    query: str
    answer: str
    mode: str
    sources: list[SourceCitation] = Field(default_factory=list)
    latency_ms: float = 0.0
    sub_queries: list[str] | None = Field(
        None, description="Sub-queries from decomposition (Phase 2b), null if not used.",
    )
    agentic_rounds: int | None = Field(
        None, description="Число раундов agentic search, null если не agentic.",
    )
    follow_up_queries: list[str] | None = Field(
        None, description="Follow-up запросы от LLM judge (agentic mode).",
    )
    agentic_round_stats: list[AgenticRoundStat] | None = Field(
        None,
        description=(
            "Per-round телеметрия agentic search (новые источники / "
            "сущности / связи + вердикт судьи). null для не-agentic."
        ),
    )
