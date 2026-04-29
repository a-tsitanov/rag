# Enterprise Knowledge Base — документация

Корпоративная база знаний на базе **LightRAG** (hybrid retrieval —
векторный поиск + knowledge graph), с FastAPI-фронтом, асинхронным
ingestion-pipeline через RabbitMQ/Taskiq и нормализацией бизнес-
идентификаторов (телефоны, ИНН, ОГРН, № договоров, адреса).

> Этот документ — единая точка входа для разработчика и оператора.
> Все команды протестированы в окружении проекта.

**Содержание**

1. [Краткое описание](#1-краткое-описание)
2. [Архитектура](#2-архитектура)
3. [Системные требования](#3-системные-требования)
4. [Установка](#4-установка)
5. [Настройка `.env`](#5-настройка-env)
6. [Инициализация и запуск](#6-инициализация-и-запуск)
7. [API: аутентификация и эндпоинты](#7-api-аутентификация-и-эндпоинты)
8. [Особенности работы](#8-особенности-работы)
9. [Эксплуатация](#9-эксплуатация)
10. [Тестирование](#10-тестирование)
11. [Безопасность](#11-безопасность)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Краткое описание

Сервис принимает документы (PDF, DOCX, PPTX, TXT, MD, EML), асинхронно
индексирует их в три хранилища и предоставляет API для семантического
поиска с генерацией ответа через LLM.

**Ключевые возможности:**

- **Hybrid retrieval**: dense (Milvus, BGE-эмбеддинги) + sparse (BM25-
  совместимый) + reranker (BGE-reranker-v2-m3). RRF-объединение.
- **Knowledge graph**: LightRAG строит граф сущностей и отношений в Neo4j,
  поддерживает 6 режимов поиска (`naive` / `local` / `global` / `hybrid`
  / `mix` / `bypass`).
- **Агентный мульти-хоп поиск**: до N раундов поиска с LLM-судьёй,
  накопление контекста, ранний exit на «нет нового», per-round телеметрия.
- **Канонизация идентификаторов**: телефоны → E.164, даты → ISO-8601, ИНН/
  ОГРН с валидацией контрольных сумм, адреса через libpostal или
  rule-based, дубли в графе схлопываются автоматически.
- **Декомпозиция запроса**: сложные вопросы разбиваются на под-запросы и
  объединяются через RRF.
- **Two-stage retrieval**: PG-prefilter по summary документа → векторный
  поиск только по релевантным.
- **Per-department access control**: фильтр по `department` на уровне
  Milvus expression.

---

## 2. Архитектура

### 2.1. Компоненты

| Сервис | Образ / Порт | Назначение |
|---|---|---|
| **api** | сборка `docker/Dockerfile`, :8000 | FastAPI: ingest + search + health |
| **worker** | сборка `docker/Dockerfile.worker` | Taskiq-обработчик ingestion-задач |
| **milvus** | `milvusdb/milvus:v2.4.17`, :19530 | Векторное хранилище (dense + sparse) |
| **etcd** | `quay.io/coreos/etcd:v3.5.16`, :2379 | Метаданные Milvus |
| **minio** | `minio/minio`, :9000 / :9001 | S3-хранилище для Milvus |
| **neo4j** | `neo4j:5-enterprise`, :7474 / :7687 | Knowledge graph (LightRAG storage) |
| **postgres** | `postgres:16-alpine`, :5432 | Статусы документов, summary, jobs |
| **rabbitmq** | `rabbitmq:3.13-management`, :5672 / :15672 | Очередь задач + dashboard |
| **ollama** | `ollama/ollama`, :11434 | Локальный LLM (через LiteLLM-прокси) |
| **litellm** | (опционально) :4000 | OpenAI-совместимый прокси для LLM |

В dev-варианте (`docker-compose.dev.yml`) дополнительно поднимается
**Attu** (`zilliz/attu`, :8001) — UI для Milvus.

### 2.2. Поток данных при ingestion

```
HTTP upload                        ┌────────────┐
   │                               │ Postgres   │
   ▼                               │ documents  │
┌──────────────┐  202 Accepted     │   row      │
│ POST /ingest │ ─────────────────►│ "pending"  │
└──────┬───────┘                   └────────────┘
       │ task in RabbitMQ                ▲
       ▼                                  │ status updates
┌──────────────────────────────────────┐  │
│ Taskiq worker (process_document)     │──┘
│  1. parse (PDF/DOCX/...)             │
│  2. chunk (semantic, ~512 токенов)   │
│  3. vectorstore: dense + sparse      │──► Milvus
│     (langchain Milvus / pymilvus)    │
│  4a. extract_identifiers (regex/lib) │
│  4b. rag.ainsert_custom_kg (canon)   │──► Neo4j (canonical nodes)
│  4c. rag.ainsert(text + augment)     │──► Neo4j (LLM-extracted entities)
│  5. PG status → "completed"          │
└──────────────────────────────────────┘
```

### 2.3. Поток данных при запросе

```
                                   ┌───────────────┐
POST /search ────► HybridSearcher  │ Milvus dense  │
                  ├──────────────► │ Milvus sparse │ ──► RRF + rerank
                  │                └───────────────┘
                  │                ┌───────────────┐
                  ├──────────────► │ Neo4j via     │
                  │                │ LightRAG      │ ──► entities + relations
                  │                └───────────────┘
                  │
                  └─► LLM (LiteLLM/Ollama) ──► answer
```

При `agentic=true` цикл повторяется до `agentic_max_rounds` раз
с LLM-судьёй между раундами; финальный синтез использует накопленные
сущности как `hl_keywords` и enriched query (см. §8.5).

---

## 3. Системные требования

**Минимум для разработки на Mac/Linux:**

- **OS**: macOS 13+ или Linux (Ubuntu 22.04+, Debian 12+).
- **Docker**: 24+, Docker Compose v2.
- **RAM**: 16 ГБ (etcd + minio + Milvus + Neo4j + Postgres + RabbitMQ +
  Ollama одновременно ≈ 6-10 ГБ; остальное — на ОС и LLM).
- **CPU**: 8+ ядер (Llama 3.3 70B на CPU неприемлема — нужен GPU или
  внешний LLM через LiteLLM).
- **Disk**: 20 ГБ под образы и data-volumes.
- **GPU** (рекомендуется): NVIDIA с поддержкой CUDA для Ollama. При
  отсутствии — работайте через внешний LLM (например, OpenAI-
  совместимый эндпоинт через LiteLLM).

**Python:** 3.12 (для локальной разработки и тестов).

**Опциональные системные пакеты:**

- **libpostal** + dev-headers — для лучшей нормализации адресов.
  Без него используется правило-основанный fallback.
  - macOS: `brew install libpostal`
  - Debian/Ubuntu: `apt-get install libpostal-dev`

---

## 4. Установка

### 4.1. Клонирование

```bash
git clone <url> enterprise-kb
cd enterprise-kb
cp .env.example .env
# отредактируйте .env (см. §5)
```

### 4.2. Через Docker Compose

**Production-like (используется в развёртывании):**

```bash
docker compose up -d
docker compose ps                # все сервисы Up?
```

**Dev-окружение** (порты bind'ятся на `127.0.0.1`, добавлен Attu UI,
понижены memory-лимиты Neo4j):

```bash
docker compose -f docker-compose.dev.yml up -d
```

Подождите 30-60 секунд пока поднимутся Milvus и Neo4j (у них есть
встроенные healthcheck'и).

### 4.3. Локальная разработка (Python)

Для запуска тестов и работы с кодом без полного Docker-стека:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**libpostal (опционально):** если нужна продакшен-уровневая нормализация
адресов:

```bash
# macOS
brew install libpostal
pip install postal

# Debian/Ubuntu
sudo apt-get install -y libpostal-dev
pip install postal
```

Без `postal` модуль `src/ingestion/identifiers.py` падает на
правило-основанный нормализатор (lowercase + раскрытие сокращений).

---

## 5. Настройка `.env`

Все настройки группируются по подсистемам через `pydantic-settings` с
префиксами. Эталон — `.env.example`. Ниже — критичные переменные.

### 5.1. API

```env
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=development              # development | staging | production
API_LOG_LEVEL=info
API_KEYS=dev-local-key,another-key   # comma-separated, см. §11
API_CORS_ORIGINS=*                # см. §11 — суз для production!
API_UPLOAD_DIR=/app/data/uploads  # SHARED volume между api и worker
```

### 5.2. RabbitMQ + Taskiq

```env
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
TASKIQ_TASK_TIMEOUT_S=1800        # hard ceiling на process_document
TASKIQ_MAX_RETRIES=2              # default_retry_count
TASKIQ_PREFETCH=2                 # сколько сообщений aio-pika забирает авансом
```

### 5.3. Milvus

```env
MILVUS_HOST=milvus                # имя сервиса в docker-compose
MILVUS_PORT=19530
MILVUS_COLLECTION=enterprise_kb
MILVUS_TIMEOUT_S=10.0
```

### 5.4. Neo4j

```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme           # ВНИМАНИЕ: смените в production!
NEO4J_TIMEOUT_S=30.0
```

### 5.5. PostgreSQL

```env
POSTGRES_DB=enterprise_kb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres        # ВНИМАНИЕ: смените в production!
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_CONNECT_TIMEOUT_S=10
```

### 5.6. LiteLLM (LLM proxy)

```env
LITELLM_BASE_URL=http://litellm:4000
LITELLM_API_KEY=sk-litellm-stub
LITELLM_MODEL=qwen2.5:3b           # модель по умолчанию
LITELLM_EMBEDDING_MODEL=nomic-embed-text
LITELLM_EMBEDDING_DIM=768
LITELLM_TIMEOUT_S=600.0
```

LiteLLM выступает прокси к Ollama, OpenAI или другим OpenAI-совместимым
бекендам. Сменить LLM — поменять `LITELLM_MODEL` и (при необходимости)
конфиг самого LiteLLM.

### 5.7. LightRAG

```env
LIGHTRAG_WORKING_DIR=/app/data/lightrag
LIGHTRAG_GRAPH_STORAGE=Neo4JStorage  # или NetworkXStorage для тестов
# Если пусто — fallback на LITELLM_MODEL / LITELLM_EMBEDDING_MODEL / DIM:
LIGHTRAG_LLM_MODEL=
LIGHTRAG_EMBEDDING_MODEL=
LIGHTRAG_EMBEDDING_DIM=0
LIGHTRAG_MAX_TOKEN_SIZE=8192
LIGHTRAG_LLM_TIMEOUT_S=600
LIGHTRAG_EMBEDDING_TIMEOUT_S=120

# Concurrency:
LIGHTRAG_MAX_ASYNC=2               # 1-2 для слабых CPU, 4+ для GPU
LIGHTRAG_NUM_CTX=16384             # КРИТИЧНО: ниже 8192 не ставить —
                                   # entity-extraction prompt не влезет
LIGHTRAG_EMBEDDING_FUNC_MAX_ASYNC=8
LIGHTRAG_EMBEDDING_BATCH_NUM=10
LIGHTRAG_MAX_PARALLEL_INSERT=2
LIGHTRAG_ENTITY_EXTRACT_MAX_GLEANING=1
```

### 5.8. Ingestion

```env
INGESTION_BATCH_SIZE=10            # batch для PG/Milvus
INGESTION_CHUNK_SIZE=512           # токенов на чанк
INGESTION_CHUNK_OVERLAP=50         # overlap между чанками
INGESTION_SUMMARY_ENABLED=true     # генерировать summary при upload?
```

---

## 6. Инициализация и запуск

### 6.1. Поднять инфраструктуру

```bash
docker compose up -d
```

### 6.2. Инициализация БД

После того как Milvus/Neo4j/PG поднялись:

```bash
# в локальном venv:
source .venv/bin/activate
python -m scripts.setup_db
```

Скрипт:
- создаёт коллекцию Milvus (`enterprise_kb`) с dense+sparse полями;
- ставит Neo4j-constraints и индексы для LightRAG;
- создаёт PG-таблицы `documents`, `chunks`.

Идемпотентно — повторный запуск не ломает существующее.

### 6.3. Запуск API и worker

В docker-compose оба запускаются автоматически. Для локальной разработки
без контейнеров (тогда `MILVUS_HOST=localhost` и т.п. в `.env`):

```bash
# API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# В отдельном терминале — worker:
taskiq worker src.ingestion.tasks:broker --workers 1
```

### 6.4. Проверка работоспособности

```bash
curl http://localhost:8000/health | jq
# {"status": "ok", "dependencies": {...}}

bash scripts/smoke.sh         # полный smoke-тест на curl
bash scripts/test_api.sh      # расширенные API-тесты (все режимы)
```

`scripts/check_ingestion.py` — диагностический отчёт по всем
backend'ам (Redis jobs, PG rows, Milvus stats, Neo4j ноды/связи,
LightRAG working_dir):

```bash
python -m scripts.check_ingestion
```

---

## 7. API: аутентификация и эндпоинты

### 7.1. Аутентификация

Все `/api/v1/*` эндпоинты требуют header `X-API-Key`. Список валидных
ключей — `API_KEYS` (comma-separated). `GET /health` — публичный.

```
401 Unauthorized — header отсутствует
403 Forbidden    — header есть, но ключ невалидный
```

### 7.2. Эндпоинты

| Method | Path | Auth | Назначение |
|---|---|---|---|
| GET | `/health` | — | Liveness + статус зависимостей |
| POST | `/api/v1/ingest` | ✅ | Загрузка документа в очередь |
| GET | `/api/v1/ingest/{job_id}` | ✅ | Статус задачи ingestion |
| POST | `/api/v1/search` | ✅ | Гибридный семантический поиск |

### 7.3. Примеры

**Health:**

```bash
curl http://localhost:8000/health
```

**Загрузка документа:**

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: dev-local-key" \
  -F "file=@./contract.pdf" \
  -F "department=legal"
# → 202 {"job_id": "uuid"}
```

**Статус задачи:**

```bash
curl http://localhost:8000/api/v1/ingest/<job_id> \
  -H "X-API-Key: dev-local-key"
# → {"status": "completed"|"processing"|"failed", "error": "..."}
```

**Поиск (hybrid mode по умолчанию):**

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "договор № ДП-2024/178-К",
    "mode": "hybrid",
    "top_k": 10,
    "department": "legal"
  }'
```

**Agentic search** (мульти-хоп):

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "кто поставщик по договору ДП-2024/178-К и какой у них ИНН?",
    "agentic": true,
    "agentic_max_rounds": 3
  }'
```

Полный набор полей `SearchRequest` — см. `src/models/search.py`.

---

## 8. Особенности работы

### 8.1. Pipeline ingestion

Реализован в `src/ingestion/worker.py` (`AsyncDocumentWorker.process_document`):

1. **Parse** (`src/ingestion/parser.py`) — PDF/DOCX/PPTX/TXT/MD/EML →
   единый текст + метаданные.
2. **Summary** (опционально) — LLM генерирует summary по первым 8 КБ
   текста, кладётся в PG для two-stage prefilter.
3. **Chunk** (`src/ingestion/chunker.py`) — `langchain SemanticChunker` с
   embedding-similarity, дефолт `chunk_size=512`, `overlap=50`.
4. **Vectorstore write** — параллельная запись dense+sparse эмбеддингов в
   Milvus через `pymilvus.MilvusClient.insert`.
5. **LightRAG entity extraction** (см. §8.2) — канонические идентификаторы
   через `ainsert_custom_kg`, затем основной `ainsert(text + augment)`.
6. **PG status** → `completed` (или `failed` с error message).

Каждый шаг тайменгуется в `ProcessResult.timings`.

### 8.2. Канонизация идентификаторов

Поверх LLM-extraction LightRAG'а работает детерминированный слой
(`src/ingestion/identifiers.py`). Зачем: одни и те же телефоны
`+7 (495) 234-56-78`, `8 495 1234567`, `+74951234567` без канонизации
создают три ноды в Neo4j вместо одной — KG-дедупликация ломается.

**Поддерживаемые типы и канонические формы:**

| Тип | Канон | Источник |
|---|---|---|
| `PhoneNumber` | E.164: `+74952345678` | google libphonenumber |
| `Email` | lowercase RFC | regex |
| `INN` | digits only, ≥10/12 знаков, valid checksum | regex + checksum |
| `OGRN` | digits only, ≥13/15 знаков, valid checksum | regex + checksum |
| `BIC` | digits only, начинается на `04` | regex |
| `ContractNumber` | uppercase, без пробелов | regex |
| `PostalAddress` | libpostal-parsed или rule-fallback | libpostal / rules |
| `DocumentDate` | ISO-8601 (YYYY-MM-DD) | dateparser |
| `Amount` | `X.XX RUB` (с поддержкой `тыс/млн/млрд`) | regex |

**Что происходит с каждым идентификатором** (Stage C):

1. `extract_identifiers(text)` находит все совпадения (с дедупом).
2. `rag.ainsert_custom_kg(payload)` создаёт **канонические** ноды в
   Neo4j ДО запуска LLM-extraction. Если такая нода уже есть из другого
   документа — descriptions конкатенируются.
3. К исходному тексту добавляется блок «Канонические идентификаторы:» с
   маппингом «оригинал → канон». LLM-extraction по этому тексту получает
   подсказку из system prompt (Stage A) использовать канон в
   `entity_name` при построении связей.

Параллельно действует Stage A — `addon_params={"entity_types": [...]}`
расширяет список типов сущностей LightRAG до бизнес-нужных
(`PhoneNumber`, `INN`, `ContractNumber` и т.д.), few-shot пример
демонстрирует канон-форматы.

### 8.3. Hybrid retrieval

`src/retrieval/hybrid_search.py` (`HybridSearcher.search`):

1. **Two-stage prefilter** (Phase 2a) — если `mode != "naive"` и в PG
   есть документы с `summary ILIKE %query%`, отсекаем по `doc_id` до
   векторного поиска.
2. **Query decomposition** (Phase 2b) — при `decompose=true` LLM режет
   запрос на под-запросы, каждый ищется параллельно, результаты RRF-
   мерджатся.
3. **Vector search** (Phase 3) — `pymilvus.hybrid_search` с
   `RRFRanker` объединяет dense + sparse результаты.
4. **Rerank** — BGE-reranker-v2-m3 пересортирует кандидатов.
5. **LightRAG synthesis** — `aquery(query, param=QueryParam(mode=...))`
   строит контекст из KG + чанков и генерирует ответ.

### 8.4. Режимы LightRAG

`mode` в `SearchRequest` определяет, как строится контекст:

| Mode | KG entities | KG relations | Vector chunks | Использование |
|---|:-:|:-:|:-:|---|
| `naive` | — | — | ✅ | Чисто косинусный поиск; baseline |
| `local` | ✅ | — | от entity | "Расскажи про X" |
| `global` | — | ✅ | от relation | "Какие тренды / связи" |
| `hybrid` | ✅ | ✅ | ✅ | local + global; default проекта |
| `mix` | ✅ | ✅ | ✅✅ | hybrid + независимый naive; **рекомендуется production** |
| `bypass` | — | — | — | Без поиска, только LLM |

### 8.5. Agentic search (мульти-хоп)

`src/retrieval/agent_search.py` (`agentic_search`). При `agentic=true`
обычный `search` заменяется на цикл:

**Раунд (до `agentic_max_rounds`):**

1. **Vector + KG search** (без LLM-синтеза) — `searcher.search(skip_rag=true)`
   и `searcher.query_graph_data` накапливают `all_sources` и
   `all_graph_data` (дедуп по `chunk_id` / `entity_name` /
   `src_id+tgt_id`).
2. **Early-exit** (Stage G): если раунд `> 1` не добавил ни одного
   нового чанка/сущности/связи — пропускаем судью, выходим из цикла.
3. **LLM-judge** (`_judge_context`): LLM оценивает накопленный контекст,
   возвращает `{sufficient, follow_up_query, reason}`. Defensive
   fallback: парсинг ошибки или exception → `sufficient=true`.
4. **Branch**: `sufficient=true` → break; иначе `current_query =
   follow_up_query`, идём в следующий раунд (защита от зацикливания:
   если `follow_up == current_query` — break).

**Финальный синтез** (Stage F):

- `enriched_query = original + "\n\nRelated sub-queries:\n- " +
  follow_ups` — расширяет keyword coverage без QueryParam-ломки.
- `hl_keywords = top-30 имён сущностей из all_graph_data` —
  пропускается через `QueryParam.hl_keywords`, минуя LightRAG'овский
  LLM keyword-extraction.
- `_ask_rag(enriched_query, hl_keywords=...)` строит финальный ответ.

**Телеметрия** (Stage H): `SearchResponse.agentic_round_stats` —
list of `AgenticRoundStat(round, query, new_sources, new_entities,
new_relations, sufficient, judge_reason)`. UI / ops видят, какой
раунд что принёс. `sufficient=null` означает skipped-judge на
early-exit.

---

## 9. Эксплуатация

### 9.1. Дедуп legacy идентификаторов

`scripts/merge_identifier_duplicates.py` — periodic merge job.
Применяет `extract_identifiers` к именам существующих нод в графе,
группирует по канонической форме, мерджит группы через
`LightRAG.amerge_entities`.

```bash
# 1. Сначала — dry-run (по умолчанию):
python -m scripts.merge_identifier_duplicates

# 2. Бэкап Neo4j:
docker compose exec neo4j neo4j-admin database dump neo4j --to-path=/data/backups

# 3. Real merge:
python -m scripts.merge_identifier_duplicates --no-dry-run

# Доп. флаги:
#   --types PhoneNumber Email   # ограничить типы
#   --limit 100                 # cap групп для частичного прохода
#   --batch-size 200            # размер батча в Neo4j get_nodes_batch
```

Запускать рекомендуется раз в неделю-сутки на cron.

### 9.2. Диагностика ingestion

```bash
python -m scripts.check_ingestion
```

Печатает:
- Redis-jobs (`job:*` хеши, статусы);
- PG `documents` (count, статусы, наличие summary);
- Milvus коллекция (rows per `doc_id`);
- Neo4j (количество Entity-нод, HAS_ENTITY-связей);
- LightRAG working_dir (размеры JSON-кэшей).

### 9.3. Eval recall идентификаторов

```bash
# Информативно (всегда exit 0):
python -m tests.eval.identifier_recall

# CI-режим (exit 1 при нарушении threshold'ов):
python -m tests.eval.identifier_recall --strict

# С JSON-выходом для machine-readable отчётов:
python -m tests.eval.identifier_recall --json-out /tmp/eval.json
```

Acceptance thresholds (см. `tests/eval/identifier_recall.py`):
- Phone/Email/INN/OGRN/BIC/Date: recall ≥ 0.95
- ContractNumber/Amount: ≥ 0.85
- PostalAddress: ≥ 0.75 (потолок без libpostal)
- Precision ≥ 0.90 везде

Чтобы расширить golden-set — добавьте JSON в
`tests/eval/golden_identifiers/` (формат смотрите в существующих файлах).

### 9.4. Логи

- **API/Worker**: `loguru` пишет в stdout. В docker — `docker compose
  logs -f api` / `... worker`.
- **Структурированные поля** в логах ingestion: `doc_id`, `path`,
  `chunks`, `lightrag` тайминг.
- **agentic_search**: per-round `sources/entities/relations` + дельты
  `(+N)`, judge `sufficient/reason/follow_up`, финал
  `hl_keywords=N latency_ms=...`.

### 9.5. RabbitMQ dashboard

`http://localhost:15672` (login `guest`/`guest` по дефолту). Видны
очередь `taskiq` и обработанные сообщения.

---

## 10. Тестирование

### 10.1. Unit + Integration (pytest)

```bash
pytest tests/                             # всё (требует Docker для test_storage)
pytest tests/test_retrieval/ -v           # retrieval
pytest tests/test_ingestion/ -v           # ingestion
pytest tests/test_scripts/ -v             # CLI-скрипты
pytest tests/eval/ -v                     # eval-gate (Stage E)
```

Текущее покрытие (на момент документа): **120 тестов в основном suite**
(retrieval + ingestion + scripts + eval), без сервисов снаружи.

### 10.2. Smoke-тесты на API

```bash
# Полный сценарий: health → upload → poll → search
bash scripts/smoke.sh

# Расширенные — все retrieval-режимы и agentic:
bash scripts/test_api.sh
```

### 10.3. Eval-gate в CI

Подключить в pipeline:

```yaml
- name: Identifier recall gate
  run: python -m tests.eval.identifier_recall --strict
```

---

## 11. Безопасность

### 11.1. API-ключи

- `API_KEYS` хранит **comma-separated** список валидных ключей.
- В **production** генерируйте 32+ символьные random-ключи (`openssl rand
  -hex 32`).
- НЕ коммитьте `.env` в репозиторий (он в `.gitignore`).

### 11.2. CORS

`API_CORS_ORIGINS=*` допустим только в development. В production —
явный список доменов:

```env
API_CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

### 11.3. Секреты в production

- Замените **обязательно**: `NEO4J_PASSWORD`, `POSTGRES_PASSWORD`,
  `LITELLM_API_KEY`, `RABBITMQ_URL` (логин/пароль).
- Не используйте дефолты `changeme`, `postgres:postgres`,
  `guest:guest`.
- Volumes с persistent data (`milvus_data`, `neo4j_data`,
  `pg_data`, `app_data`) — обеспечьте регулярный backup.

### 11.4. Сетевая изоляция

В `docker-compose.yml` все сервисы общаются через внутреннюю сеть
`enterprise-kb_default`. Наружу пробрасывайте только `api:8000`. В
dev-варианте порты bind'ятся на `127.0.0.1` (предотвращает доступ из
LAN).

---

## 12. Troubleshooting

### LightRAG: «num_ctx is too small»

Симптом: ошибки в логах ingestion при entity-extraction.

Причина: дефолтный Ollama-`num_ctx=2048` режет system prompt + chunk
LightRAG.

Фикс: `LIGHTRAG_NUM_CTX=16384` (или больше) в `.env`. Не ставьте ниже
8192.

### `postal` не импортируется

Симптом: `ModuleNotFoundError: No module named 'postal'`.

Причина: `pip install postal` требует системную libpostal C-либу.

Фикс: либо установите libpostal (см. §3), либо игнорируйте — модуль
`identifiers.py` импортирует postal через `try/except ImportError` и
без него работает на rule-fallback'е.

### LLM-таймауты при ingestion больших документов

Симптом: `LIGHTRAG_LLM_TIMEOUT` exceeded в логах.

Причина: длинные чанки + слабый LLM.

Фикс:
1. Уменьшить `INGESTION_CHUNK_SIZE` (с 512 до 384).
2. Увеличить `LIGHTRAG_LLM_TIMEOUT_S` (600 → 1200).
3. Уменьшить `LIGHTRAG_MAX_ASYNC` (2 → 1) при перегрузке GPU.

### Milvus / Neo4j не поднимаются

```bash
docker compose ps              # health-checks
docker compose logs milvus
docker compose logs neo4j
```

Проверьте свободное место на диске (`docker system df`) и память
(macOS: Docker Desktop → Settings → Resources, выделить 8+ ГБ).

### `setup_db.py` падает с ошибкой подключения

Запускайте после того, как сервисы прошли healthcheck:

```bash
docker compose up -d
sleep 60                       # дождаться Milvus + Neo4j
python -m scripts.setup_db
```

### Agentic search «зависает» на 3 раунда

Включите DEBUG-логи (`API_LOG_LEVEL=debug`) — `agent_search` пишет
per-round дельты `(+N)` и judge-вердикт. Если каждый раунд приносит
новое — это корректное поведение, увеличьте `agentic_max_rounds=5` или
снизьте.

### Идентификаторы дублируются в Neo4j

Запустите Stage-D job:

```bash
python -m scripts.merge_identifier_duplicates           # dry-run
python -m scripts.merge_identifier_duplicates --no-dry-run
```

Если дубликаты не схлопываются — проверьте, что у нод стоит
`entity_type` (старые ноды до Stage A могли его не иметь):

```cypher
MATCH (n) WHERE n.entity_type IS NULL RETURN COUNT(n);
```

---

## Дополнительно

- **Архитектурные решения** про канонизацию и agentic-search закреплены
  в `~/.claude/plans/hashed-rolling-llama.md` и memory-файлах
  `~/.claude/projects/.../memory/`.
- **Roadmap** (out of scope текущего плана): фильтры
  department/doc_type в графовом слое (требует per-department
  collections в LightRAG), ФИАС-дамп для адресов, переход финального
  синтеза `agentic_search` на `mode=bypass` с ручной prompt-сборкой.
