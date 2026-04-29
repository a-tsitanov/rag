"""LightRAG factory + graph-storage env shims.

Public API is now a single pure factory :func:`create_rag` — it returns a
fresh :class:`lightrag.LightRAG` instance without touching any module
globals.  Lifecycle (teardown) is the caller's responsibility (the dishka
provider does it via ``yield`` + ``close_rag_graph``).

The old singleton triad (``init_rag`` / ``get_rag`` / ``close_rag``) was
removed — a container now owns the instance.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


# ── domain-specific entity types & prompt customization ───────────────
#
# LightRAG defaults (lightrag/constants.py:DEFAULT_ENTITY_TYPES) are
# Person/Organization/Location/Event/Concept/...  Identifiers (phones,
# INN, OGRN, contract numbers, addresses, amounts, dates) fall through
# to "Other" or get dropped.  The list below is tuned for Russian
# business documents.
BUSINESS_ENTITY_TYPES: list[str] = [
    "Person",
    "Organization",
    "Location",
    "PhoneNumber",
    "Email",
    "ContractNumber",
    "OrderNumber",
    "InvoiceNumber",
    "PostalAddress",
    "INN",
    "OGRN",
    "BIC",
    "BankAccount",
    "DocumentDate",
    "Amount",
    "Concept",
    "Method",
    "Event",
]


# Extra rules appended to the default extraction system prompt (rule №9).
# Teaches the LLM (a) to extract identifiers verbatim when they appear
# raw in text, and (b) to switch to canonical form when an augment-block
# "Канонические идентификаторы" is supplied by the ingestion pipeline
# (see Stage C of the retrieval-quality plan).
_IDENTIFIER_RULES = """

9.  **Identifier Extraction (Critical):**
    *   Entity types that represent identifiers — `PhoneNumber`, `Email`,
        `ContractNumber`, `OrderNumber`, `InvoiceNumber`, `PostalAddress`,
        `INN`, `OGRN`, `BIC`, `BankAccount`, `DocumentDate`, `Amount` —
        must be extracted whenever they appear, even if they look trivial.
    *   **Canonical-form rule:** if the input contains a section titled
        "Канонические идентификаторы" (canonical identifiers block), use
        those exact strings as `entity_name` for matching identifiers.
        Otherwise extract the identifier **verbatim** as it appears in the
        source — preserve all separators (`-`, `/`, `.`, spaces,
        parentheses, `№`).  Do NOT normalize, abbreviate, or reformat.
        *   Wrong: `+7 (495) 123-45-67` → `74951234567` (when no canonical
            block is given).
        *   Right: `+7 (495) 123-45-67` → `+7 (495) 123-45-67`.
    *   `entity_description` for identifiers must include the role/owner
        and the original written form when a canonical was used.  Example:
        `entity_description = "Контактный телефон ООО Ромашка по договору
        №12-345/АВ; в тексте: '+7 (495) 123-45-67'"`.
    *   Always create a relationship from the identifier-owning entity to
        the identifier itself (e.g. `Person → PhoneNumber`,
        `Organization → INN`, `ContractNumber → DocumentDate`).
"""


# One Russian-domain few-shot replaces the three default story-style
# examples.  Goal: teach the model the identifier-extraction pattern,
# including how to use a "Канонические идентификаторы" block when it is
# supplied.  `{tuple_delimiter}` and `{completion_delimiter}` are
# placeholders — LightRAG's `.format()` substitutes them at runtime
# (see lightrag/operate.py).
_BUSINESS_DOC_EXAMPLE = """<Entity_types>
["Person","Organization","Location","PhoneNumber","Email","ContractNumber","PostalAddress","INN","Amount","DocumentDate"]

<Input Text>
```
Договор поставки №ДП-2024/178-К от 15.03.2024 заключён между ООО «Северные технологии» (ИНН 7707083893, юр. адрес: 127015, г. Москва, ул. Бутырская, д. 76, стр. 1) и АО «Промсервис».

Контактное лицо: Иванов Иван Петрович, телефон +7 (495) 234-56-78, доб. 412, e-mail: i.ivanov@sevtech.ru. Сумма договора: 4 250 000,00 руб.

---
Канонические идентификаторы (используй ИМЕННО ТАКУЮ форму в entity_name):
- PhoneNumber: +74952345678
- Email: i.ivanov@sevtech.ru
- INN: 7707083893
- ContractNumber: ДП-2024/178-К
- DocumentDate: 2024-03-15
- Amount: 4250000.00 RUB
- PostalAddress: 127015, москва, ул бутырская, 76 стр 1
```

<Output>
entity{tuple_delimiter}ДП-2024/178-К{tuple_delimiter}ContractNumber{tuple_delimiter}Номер договора поставки между ООО «Северные технологии» и АО «Промсервис», заключённого 15.03.2024.
entity{tuple_delimiter}2024-03-15{tuple_delimiter}DocumentDate{tuple_delimiter}Дата заключения договора поставки №ДП-2024/178-К; в тексте: «15.03.2024».
entity{tuple_delimiter}ООО «Северные технологии»{tuple_delimiter}Organization{tuple_delimiter}Юридическое лицо — поставщик по договору №ДП-2024/178-К.
entity{tuple_delimiter}АО «Промсервис»{tuple_delimiter}Organization{tuple_delimiter}Юридическое лицо — покупатель по договору №ДП-2024/178-К.
entity{tuple_delimiter}7707083893{tuple_delimiter}INN{tuple_delimiter}ИНН ООО «Северные технологии».
entity{tuple_delimiter}127015, москва, ул бутырская, 76 стр 1{tuple_delimiter}PostalAddress{tuple_delimiter}Юридический адрес ООО «Северные технологии»; в тексте: «127015, г. Москва, ул. Бутырская, д. 76, стр. 1».
entity{tuple_delimiter}Иванов Иван Петрович{tuple_delimiter}Person{tuple_delimiter}Контактное лицо со стороны ООО «Северные технологии» по договору №ДП-2024/178-К.
entity{tuple_delimiter}+74952345678{tuple_delimiter}PhoneNumber{tuple_delimiter}Контактный телефон Иванова И.П.; в тексте: «+7 (495) 234-56-78, доб. 412».
entity{tuple_delimiter}i.ivanov@sevtech.ru{tuple_delimiter}Email{tuple_delimiter}Рабочий e-mail Иванова И.П. в ООО «Северные технологии».
entity{tuple_delimiter}4250000.00 RUB{tuple_delimiter}Amount{tuple_delimiter}Сумма договора поставки №ДП-2024/178-К; в тексте: «4 250 000,00 руб.».
relation{tuple_delimiter}ООО «Северные технологии»{tuple_delimiter}АО «Промсервис»{tuple_delimiter}поставка, договор{tuple_delimiter}Организации связаны договором поставки №ДП-2024/178-К от 15.03.2024.
relation{tuple_delimiter}ООО «Северные технологии»{tuple_delimiter}ДП-2024/178-К{tuple_delimiter}участник договора{tuple_delimiter}ООО «Северные технологии» — поставщик по этому договору.
relation{tuple_delimiter}АО «Промсервис»{tuple_delimiter}ДП-2024/178-К{tuple_delimiter}участник договора{tuple_delimiter}АО «Промсервис» — покупатель по этому договору.
relation{tuple_delimiter}ДП-2024/178-К{tuple_delimiter}2024-03-15{tuple_delimiter}дата заключения{tuple_delimiter}Договор заключён 15 марта 2024 года.
relation{tuple_delimiter}ООО «Северные технологии»{tuple_delimiter}7707083893{tuple_delimiter}реквизит, идентификация{tuple_delimiter}ИНН принадлежит ООО «Северные технологии».
relation{tuple_delimiter}ООО «Северные технологии»{tuple_delimiter}127015, москва, ул бутырская, 76 стр 1{tuple_delimiter}юридический адрес{tuple_delimiter}Адрес регистрации ООО «Северные технологии».
relation{tuple_delimiter}Иванов Иван Петрович{tuple_delimiter}ООО «Северные технологии»{tuple_delimiter}сотрудник, контакт{tuple_delimiter}Иванов И.П. — контактное лицо ООО «Северные технологии» по данному договору.
relation{tuple_delimiter}Иванов Иван Петрович{tuple_delimiter}+74952345678{tuple_delimiter}контакт{tuple_delimiter}Указанный телефон принадлежит Иванову И.П.
relation{tuple_delimiter}Иванов Иван Петрович{tuple_delimiter}i.ivanov@sevtech.ru{tuple_delimiter}контакт{tuple_delimiter}Указанный e-mail принадлежит Иванову И.П.
relation{tuple_delimiter}ДП-2024/178-К{tuple_delimiter}4250000.00 RUB{tuple_delimiter}сумма договора{tuple_delimiter}Сумма обязательств по договору поставки.
{completion_delimiter}

"""


_PROMPTS_PATCHED = False


def _patch_lightrag_prompts() -> None:
    """Mutate ``lightrag.prompt.PROMPTS`` with project-specific overrides.

    Idempotent — safe to call multiple times (e.g. across multiple
    ``create_rag`` invocations in tests).  Must run **before**
    ``LightRAG(...)`` is constructed since LightRAG snapshots prompt
    references at init time.
    """
    global _PROMPTS_PATCHED
    if _PROMPTS_PATCHED:
        return

    from lightrag import prompt as lr_prompt

    base_system = lr_prompt.PROMPTS["entity_extraction_system_prompt"]
    if "Identifier Extraction (Critical)" not in base_system:
        lr_prompt.PROMPTS["entity_extraction_system_prompt"] = base_system.replace(
            "\n---Examples---\n{examples}",
            _IDENTIFIER_RULES + "\n---Examples---\n{examples}",
        )

    lr_prompt.PROMPTS["entity_extraction_examples"] = [_BUSINESS_DOC_EXAMPLE]

    _PROMPTS_PATCHED = True
    logger.info(
        "LightRAG prompts patched  identifier_rules=on  few_shot=business_doc"
    )


# ── env-var bridges for LightRAG storage backends ────────────────────


def _export_neo4j_env() -> None:
    """``Neo4JStorage`` в LightRAG 1.4 читает креды **только** из
    ``os.environ`` — конструктор-kwargs для URI/USERNAME/PASSWORD нет.

    Используем прямое присвоение (не ``setdefault``), чтобы наши
    значения из ``Settings`` перекрывали возможные устаревшие env-vars
    от предыдущих процессов / docker-переопределений.
    """
    os.environ["NEO4J_URI"] = settings.neo4j.uri
    os.environ["NEO4J_USERNAME"] = settings.neo4j.user
    os.environ["NEO4J_PASSWORD"] = settings.neo4j.password


def _export_milvus_env() -> None:
    os.environ["MILVUS_URI"] = (
        f"http://{settings.milvus.host}:{settings.milvus.port}"
    )


# ── factory ───────────────────────────────────────────────────────────


async def create_rag(
    *,
    working_dir: str | None = None,
    llm_model_name: str | None = None,
    embed_model: str | None = None,
    embedding_dim: int | None = None,
    max_token_size: int | None = None,
    graph_storage: str | None = None,
):
    """Build and return a fresh LightRAG instance.

    All model/dim args default to the resolved values in
    :class:`~src.config.Settings` — set
    ``LIGHTRAG_LLM_MODEL`` / ``LIGHTRAG_EMBEDDING_MODEL`` /
    ``LIGHTRAG_EMBEDDING_DIM`` in the environment to override, or leave
    them empty to reuse ``OLLAMA_MODEL`` / ``EMBEDDING_MODEL`` /
    ``EMBEDDING_DIM``.

    Caller owns teardown.  For the Neo4j backend call
    :func:`close_rag_graph` on the returned instance to close the driver.
    """
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    _export_neo4j_env()
    _export_milvus_env()
    _patch_lightrag_prompts()

    wd = working_dir or settings.lightrag.working_dir
    llm_name = llm_model_name or settings.effective_lightrag_llm_model
    embed_name = embed_model or settings.effective_lightrag_embedding_model
    embed_dim = embedding_dim or settings.effective_lightrag_embedding_dim
    max_tok = max_token_size or settings.lightrag.max_token_size
    graph_kind = graph_storage or settings.lightrag.graph_storage

    Path(wd).mkdir(parents=True, exist_ok=True)

    # ── OpenAI-compatible backend (LiteLLM proxy) ─────────────────────
    from lightrag.llm.openai import openai_complete, openai_embed

    _openai_embed_raw = getattr(openai_embed, "func", openai_embed)

    async def _embed(texts: list[str]) -> list:
        return await _openai_embed_raw(
            texts,
            model=embed_name,
            base_url=settings.litellm.base_url,
            api_key=settings.litellm.api_key,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=embed_dim,
        max_token_size=max_tok,
        func=_embed,
    )

    rag = LightRAG(
        working_dir=wd,
        llm_model_func=openai_complete,
        llm_model_name=llm_name,
        embedding_func=embedding_func,
        graph_storage=graph_kind,
        default_llm_timeout=settings.lightrag.llm_timeout_s,
        default_embedding_timeout=settings.lightrag.embedding_timeout_s,
        # ── concurrency knobs (see LightRAGSettings for rationale) ───
        llm_model_max_async=settings.lightrag.max_async,
        embedding_func_max_async=settings.lightrag.embedding_func_max_async,
        embedding_batch_num=settings.lightrag.embedding_batch_num,
        max_parallel_insert=settings.lightrag.max_parallel_insert,
        entity_extract_max_gleaning=settings.lightrag.entity_extract_max_gleaning,
        llm_model_kwargs={
            "base_url": settings.litellm.base_url,
            "api_key": settings.litellm.api_key,
            # Для downstream-Ollama LiteLLM пробрасывает extra_body.options
            # как ollama options.  Дефолтный num_ctx=2048 режет LightRAG
            # entity-extraction prompt (system_prompt + chunk > 2k tok).
            # Не-Ollama backends значение игнорируют.
            "extra_body": {"options": {"num_ctx": settings.lightrag.num_ctx}},
        },
        addon_params={
            "language": "Russian",
            "entity_types": BUSINESS_ENTITY_TYPES,
        },
    )

    # LightRAG 1.4+ requires explicit storage initialization before any
    # ainsert / aquery call.  It populates pipeline_status and opens the
    # graph driver.
    await rag.initialize_storages()

    logger.info(
        "LightRAG created  working_dir=%s  llm=%s (timeout=%ds, max_async=%d)  "
        "embed=%s (dim=%d, timeout=%ds)  max_tokens=%d  graph=%s",
        wd, llm_name,
        settings.lightrag.llm_timeout_s, settings.lightrag.max_async,
        embed_name, embed_dim, settings.lightrag.embedding_timeout_s,
        max_tok, graph_kind,
    )
    return rag


async def close_rag_graph(rag) -> None:
    """Best-effort shutdown of LightRAG's graph driver (Neo4j).

    NetworkX graphs write to disk on ``index_done_callback`` — no close
    needed. Neo4j keeps a driver open that must be closed explicitly.
    """
    try:
        graph = getattr(rag, "chunk_entity_relation_graph", None)
        if graph is not None and hasattr(graph, "close"):
            await graph.close()
    except Exception as exc:
        logger.warning("error closing LightRAG graph: %s", exc)
