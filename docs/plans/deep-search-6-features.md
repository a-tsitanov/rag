# Plan: глубокий поиск — 6 фич, максимально простой подход

## Context

Текущий search pipeline: один `vectorstore.similarity_search_with_score` +
один `LightRAG.aquery` → ответ. На масштабе >1000 документов этого
недостаточно: нет keyword-поиска, нет фильтрации по дате/типу, нет
multi-hop reasoning.

Пользователь хочет реализовать **все 6 паттернов** из обзора, но
**максимально просто** — без новой инфры, с минимальным новым кодом.

4 фазы, каждая самоценна (можно остановиться после любой):

---

## Phase 1 — Foundation (фильтры + саммари + QueryParam knobs)

### 1a. Metadata-фильтры в search endpoint

**Что:** добавить `created_after`, `created_before`, `doc_type_filter` в
`SearchRequest`. Формируют Milvus `expr` (уже поддержан).

**Файлы:**
- `src/models/search.py` — новые optional поля в `SearchRequest`
- `src/retrieval/hybrid_search.py` — строить `expr` из фильтров

```python
class SearchRequest(BaseModel):
    ...
    created_after: int | None = None    # unix timestamp
    created_before: int | None = None
    doc_type_filter: str | None = None  # "pdf", "docx", etc.
```

В `HybridSearcher.search`:
```python
parts = []
if department: parts.append(f'department == "{department}"')
if doc_type_filter: parts.append(f'doc_type == "{doc_type_filter}"')
if created_after: parts.append(f'created_at >= {created_after}')
if created_before: parts.append(f'created_at <= {created_before}')
expr = " and ".join(parts) if parts else None
```

### 1b. Полный набор QueryParam knobs

**Что:** прокинуть `top_k`, `chunk_top_k`, `max_entity_tokens`,
`max_relation_tokens`, `max_total_tokens` из `SearchRequest` в
`QueryParam` LightRAG. Дефолты = LightRAG'овские.

**Файлы:**
- `src/models/search.py` — новые поля с дефолтами
- `src/retrieval/hybrid_search.py::_ask_rag` — прокинуть все knobs

### 1c. Document summaries при ingestion

**Что:** при ingestion вызвать LLM один раз: "Summarize this document
in 2-3 sentences". Результат → `documents.summary TEXT` в Postgres.

**Файлы:**
- `scripts/setup_db.py` — `ALTER TABLE documents ADD COLUMN summary TEXT`
- `src/ingestion/worker.py` — после parse, до chunk: вызов
  `ollama.chat(model=settings.ollama.model, messages=[...])` с
  промптом на саммаризацию. Записать в `PGStatusPayload.summary` (новое поле).
- `src/di/providers.py` — `_UPSERT_DOC` SQL: добавить `summary` колонку
- `src/config.py` — `IngestionSettings.summary_enabled: bool = True`,
  `IngestionSettings.summary_prompt: str = "..."`

**Простота:** один дополнительный LLM-вызов на документ. Промпт
фиксированный (не кастомный per-doc).

---

## Phase 2 — Smart search (декомпозиция запроса + двухэтапный поиск)

### 2a. Hierarchical two-stage search

**Что:** при `mode != "naive"`, сначала ищем по PG `documents.summary`
(full-text или LIKE), получаем top-N `doc_id`'ов, потом ищем чанки
в Milvus **только внутри этих doc_id** через `expr`.

```python
# stage 1: find relevant documents by summary
doc_ids = await pg.execute(
    "SELECT id FROM documents WHERE summary ILIKE $1 LIMIT $2",
    (f"%{query}%", 20)
)

# stage 2: vector search only within those documents
expr = f'doc_id in {[str(d) for d in doc_ids]}'
results = vectorstore.similarity_search_with_score(query, k=top_k, expr=expr)
```

**Файлы:**
- `src/retrieval/hybrid_search.py` — новый метод `_two_stage_search`
- Fallback: если stage-1 вернул 0 doc_ids → обычный полный поиск.

**Простота:** Postgres `ILIKE` по `summary` — не нужен FTS-индекс для
первой версии. Позже можно добавить `tsvector` + GIN.

### 2b. Query decomposition

**Что:** перед поиском отправить запрос LLM:
"Break this query into 1-3 independent sub-queries. Return JSON array."
Для каждого sub-query запустить поиск, merge результаты через RRF
(Reciprocal Rank Fusion).

**Файлы:**
- `src/retrieval/query_decomposer.py` — **новый файл**, ~40 строк:
  ```python
  async def decompose_query(query: str, ollama: AsyncClient) -> list[str]:
      resp = await ollama.chat(model=settings.ollama.model, messages=[
          {"role": "system", "content": DECOMPOSE_PROMPT},
          {"role": "user", "content": query},
      ])
      return json.loads(resp["message"]["content"])  # ["sub-q1", "sub-q2"]
  ```
- `src/retrieval/hybrid_search.py` — в `search()`: если включена
  декомпозиция, вызвать `decompose_query`, запустить N поисков
  параллельно через `asyncio.gather`, merge через RRF.
- `src/models/search.py` — `SearchRequest.decompose: bool = False`

**RRF merge** — ~10 строк:
```python
def rrf_merge(results_lists, k=60):
    scores = defaultdict(float)
    for results in results_lists:
        for rank, (doc, score) in enumerate(results):
            scores[doc.page_content] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## Phase 3 — Hybrid retrieval (BM25 sparse via Milvus)

**Что:** добавить второе vector-поле `sparse_embedding
(SPARSE_FLOAT_VECTOR)` в коллекцию Milvus. При ingestion считать BM25
sparse-вектора через `pymilvus.model.sparse.BM25EmbeddingFunction`.
При search — hybrid search (dense + sparse) с `RRFRanker`.

**Ограничение:** BM25EmbeddingFunction нужен fitted corpus (`.fit(docs)`).
Corpus пересчитывается при каждом рестарте worker'а или кешируется.

**Файлы:**
- `src/storage/bm25_model.py` — **новый файл**: загрузка/сохранение
  fitted BM25 модели (pickle в `./data/bm25_model.pkl`).
  При первом запуске — fit на текущих чанках из Milvus/PG.
- `scripts/setup_db.py` — пересоздать коллекцию с новым sparse-полем
  (миграция: drop + recreate).
- `src/ingestion/worker.py` — при Milvus write добавлять
  `sparse_embedding` в metadata.
- `src/retrieval/hybrid_search.py` — использовать
  `Collection.hybrid_search([dense_req, sparse_req], RRFRanker())` вместо
  `similarity_search_with_score`. Это **обход langchain** для search
  (langchain-milvus пока не поддерживает hybrid multi-vector).
  Ingestion через langchain остаётся.

**Простота:** BM25EmbeddingFunction из pymilvus делает всё внутри, нам
только `.fit()` и `.encode_documents()` / `.encode_queries()`.

---

## Phase 4 — Agentic RAG (итеративный поиск)

**Что:** после первого раунда поиска, LLM оценивает: "Достаточно ли
контекста для ответа?" Если нет — генерирует follow-up query и
ищет ещё раз. Максимум 3 итерации.

**Файлы:**
- `src/retrieval/agent_search.py` — **новый файл**, ~60 строк:
  ```python
  async def agentic_search(
      searcher: HybridSearcher,
      query: str,
      ollama: AsyncClient,
      max_rounds: int = 3,
  ) -> SearchResponse:
      all_sources = []
      current_query = query
      
      for _ in range(max_rounds):
          resp = await searcher.search(current_query, ...)
          all_sources.extend(resp.sources)
          
          # ask LLM: enough context?
          judgment = await ollama.chat(messages=[
              {"role": "system", "content": JUDGE_PROMPT},
              {"role": "user", "content": f"Query: {query}\nContext: {_context(all_sources)}"},
          ])
          
          parsed = json.loads(judgment.message.content)
          if parsed["sufficient"]:
              break
          current_query = parsed["follow_up_query"]
      
      # final answer with all accumulated sources
      return await searcher.search(query, ..., extra_context=all_sources)
  ```
- `src/models/search.py` — `SearchRequest.agentic: bool = False`
- `src/api/routes/search.py` — если `req.agentic`, вызвать
  `agentic_search` вместо прямого `searcher.search`.

**Простота:** while-loop + один LLM-вызов per iteration для judgment.
Не нужен LangGraph/CrewAI — plain Python.

---

## Execution order

1. Phase 1a (metadata filters) → 1b (QueryParam) → 1c (summaries)
2. Phase 2a (two-stage) → 2b (decomposition)
3. Phase 3 (BM25 sparse)
4. Phase 4 (agentic)

Каждый шаг самоценен и тестируем отдельно.

## Critical files to modify

| File | Phases |
|---|---|
| `src/models/search.py` | 1a, 1b, 2b, 4 |
| `src/retrieval/hybrid_search.py` | 1a, 1b, 2a, 3 |
| `src/ingestion/worker.py` | 1c, 3 |
| `src/di/providers.py` | 1c, 3 |
| `scripts/setup_db.py` | 1c, 3 |

## New files

| File | Phase |
|---|---|
| `src/retrieval/query_decomposer.py` | 2b |
| `src/retrieval/agent_search.py` | 4 |
| `src/storage/bm25_model.py` | 3 |

## Verification per phase

**Phase 1:**
```bash
# metadata filters
curl -X POST /api/v1/search -d '{"query":"...", "doc_type_filter":"pdf", "created_after": 1700000000}'
# summaries
curl /api/v1/ingest/{id}  # → status=completed, documents.summary заполнен
SELECT summary FROM documents WHERE id = '...'
```

**Phase 2:**
```bash
# decomposition
curl -X POST /api/v1/search -d '{"query":"How does K8s affect budget across depts?", "decompose": true}'
# → logs покажут sub-queries
```

**Phase 3:**
```bash
# hybrid: keyword "Kubernetes" + semantic "container orchestration"
curl -X POST /api/v1/search -d '{"query":"K8s CNI plugin", "mode":"hybrid"}'
# → BM25 поймает точный термин "CNI" даже если embedding его не видит
```

**Phase 4:**
```bash
curl -X POST /api/v1/search -d '{"query":"complex multi-hop question", "agentic": true}'
# → logs покажут N rounds, follow-up queries
```
