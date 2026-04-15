# Enterprise KB — структура проекта и стек

*Снимок на 2026-04-15. Актуальные версии — из `requirements.txt`, текущая топология — из `docker-compose.yml`.*

## Стек

### Инфраструктура (docker compose)

| Сервис | Образ | Порт(ы) | Назначение |
|---|---|---|---|
| **rabbitmq** | `rabbitmq:3.13-management` | 5672 AMQP / **15672 UI** | Брокер задач + dashboard |
| **milvus** | `milvusdb/milvus:v2.4.17` | 19530, 9091 | Vector DB (HNSW, COSINE) |
| **etcd** | `quay.io/coreos/etcd:v3.5.16` | — | Metadata для Milvus |
| **minio** | `minio/minio:latest` | 9000, 9001 | Object storage для Milvus |
| **neo4j** | `neo4j:5-enterprise` | 7474, 7687 | Knowledge graph (APOC + GDS) |
| **postgres** | `postgres:16-alpine` | 5432 | Статус документов, чанки |
| **ollama** | `ollama/ollama:latest` | 11434 | Embeddings + LLM |
| **api** | локальный `docker/Dockerfile` | 8000 | FastAPI |
| **worker** | локальный `docker/Dockerfile.worker` | — | `taskiq worker` |

### Приложение (Python 3.12, pinned versions)

| Слой | Библиотека | Версия |
|---|---|---|
| Web / API | `fastapi` + `uvicorn[standard]` | 0.135.3 / 0.44.0 |
| DI | `dishka` | 1.9.1 |
| RAG | `lightrag-hku` | 1.4.14 |
| Vector DB | `pymilvus` (+ `milvus-lite` для тестов) | 2.6.12 / 2.5.1 |
| Graph DB | `neo4j` | 6.1.0 |
| RDBMS | `psycopg[binary]` | 3.3.3 |
| Queue | `taskiq` + `taskiq-aio-pika` + `aio-pika` | 0.12.1 / 0.6.0 / 9.6.2 |
| Retry | `tenacity` | 9.1.4 |
| Валидация | `pydantic` + `pydantic-settings` | 2.13.0 / 2.13.1 |
| Логи | `structlog` | 25.5.0 |
| LLM client | `ollama` + `openai` | latest / 2.31.0 |
| Парсинг | `pypdf`, `pdfplumber`, `python-docx`, `python-pptx`, `beautifulsoup4` | 6.10.1 / 0.11.9 / 1.2.0 / 1.0.2 / 4.12.3 |
| NLP | `nltk`, `tiktoken`, `langchain-text-splitters` | 3.9.4 / 0.12.0 / 0.3.4 |
| Тесты | `pytest`, `pytest-asyncio`, `httpx`, `testcontainers`, `ruff` | 9.0.3 / 1.3.0 / 0.28.1 / 4.14.2 / 0.8.6 |

## Структура каталогов

```
enterprise-kb/
├── CLAUDE.md                     # контракт для Claude Code
├── STRUCTURE.md                  # этот файл
├── docker-compose.yml            # 8 сервисов, 8 именованных volumes
├── requirements.txt              # 3.12 + pinned deps
├── .env, .env.example            # RABBITMQ_URL, NEO4J_*, POSTGRES_*, OLLAMA_*, API_KEYS
│
├── docker/
│   ├── Dockerfile                # API: uvicorn src.api.main:app
│   └── Dockerfile.worker         # Worker: taskiq worker src.ingestion.tasks:broker
│
├── scripts/
│   ├── init_db.py
│   ├── setup_db.py               # Milvus collection, Neo4j constraints, PG schema+миграции
│   └── start.sh
│
├── src/
│   ├── main.py
│   ├── config.py                 # pydantic-settings (Settings + singleton settings)
│   │
│   ├── di/                       # 🔑 Единая точка инициализации клиентов
│   │   ├── __init__.py           # build_api_container / build_worker_container
│   │   └── providers.py          # CommonProvider / ApiProvider / WorkerProvider
│   │
│   ├── api/
│   │   ├── main.py               # FastAPI app + lifespan (брокер + dishka)
│   │   ├── auth.py               # X-API-Key dependency
│   │   ├── middleware.py         # structlog + request_id
│   │   └── routes/
│   │       ├── health.py         # /health (RabbitMQ / Milvus / Neo4j / LightRAG)
│   │       ├── search.py         # POST /api/v1/search
│   │       └── ingest.py         # POST /api/v1/ingest + GET /api/v1/ingest/{job_id}
│   │
│   ├── ingestion/
│   │   ├── parser.py             # DocumentParser (PDF/DOCX/PPTX/TXT/MD/EML)
│   │   ├── chunker.py            # SemanticChunker (async, BGE-M3 similarity)
│   │   ├── worker.py             # AsyncDocumentWorker (6 этапов pipeline) + BatchProcessor
│   │   └── tasks.py              # 🔑 AioPikaBroker + @broker.task process_document
│   │
│   ├── retrieval/
│   │   ├── lightrag_setup.py     # create_rag() / close_rag_graph()
│   │   └── hybrid_search.py      # HybridSearcher (Milvus candidates → rerank → LightRAG)
│   │
│   ├── storage/
│   │   ├── milvus_client.py      # AsyncMilvusClient (HNSW M=16 / efC=200)
│   │   └── neo4j_client.py       # AsyncNeo4jClient (pool=50, APOC)
│   │
│   ├── models/
│   │   ├── document.py
│   │   └── search.py
│   │
│   ├── core/embeddings.py
│   ├── utils/logging.py          # structlog JSON configure_logging
│   └── services/__init__.py      # legacy, пусто
│
└── tests/
    ├── conftest.py
    ├── test_api/test_health.py   # ⚠ pre-existing broken
    ├── test_ingestion/
    │   ├── fixtures/             # sample.{pdf,docx,pptx,txt,md,eml}
    │   ├── test_parser.py        # 6 тестов
    │   ├── test_chunker.py       # 8 тестов (async)
    │   ├── test_queue.py         # 3 теста (taskiq InMemoryBroker)
    │   └── test_worker.py        # 7 тестов (fake backends)
    ├── test_retrieval/
    │   ├── test_lightrag_setup.py
    │   └── test_hybrid_search.py
    └── test_storage/
        ├── test_milvus_client.py
        └── test_neo4j_client.py
```

## Ключевые точки входа

| Что | Команда / Путь |
|---|---|
| API | `uvicorn src.api.main:app --port 8000` |
| Worker | `taskiq worker src.ingestion.tasks:broker --workers 1` |
| DI-контейнеры | `src.di.build_api_container()` / `build_worker_container()` |
| Брокер (singleton) | `src.ingestion.tasks.broker` |
| Основная задача | `src.ingestion.tasks.process_document` (kicked as `.kiq(...)`) |
| Pipeline ядро | `src.ingestion.worker.AsyncDocumentWorker.process_document` |
| Поиск | `src.retrieval.hybrid_search.HybridSearcher.search` |
| DB setup | `python scripts/setup_db.py` (идемпотентно + миграции) |

## Потоки данных

### Ingestion
```
POST /ingest → UUID doc_id → INSERT documents(pending)
     → process_document.kiq(doc_id, path, department, priority)
     → RabbitMQ (priority queue, max_priority=10)
     → taskiq worker → AsyncDocumentWorker.process_document
         ├─ parse → chunk → embed (Ollama BGE-M3)
         ├─ LightRAG.ainsert (→ Neo4j graph storage)
         ├─ Milvus upsert
         └─ UPDATE documents(processing → completed | failed)
     retry: SimpleRetryMiddleware (3×) → RabbitMQ DLX at terminal failure
```

### Retrieval (`/api/v1/search`)
```
embed(query) → Milvus.search(top_k*3, department filter)
             → rerank (stub / BGE-reranker-v2-m3)   ─┐
             → LightRAG.aquery(mode=hybrid)         ─┤ asyncio.gather
                                                     ▼
                             SearchResponse(answer, sources, latency_ms)
```

## Что разведено и почему

- **LightRAG graph storage**: API использует `NetworkXStorage` (in-process, чтобы не гонялся с воркером), worker — `Neo4JStorage` (общий графовый backend для N реплик воркеров).
- **RabbitMQ priority queue**: `low/normal/high` → `0/5/9`; маппинг в `src.ingestion.tasks.priority_value`.
- **Источник правды статуса**: Postgres `documents` (id, path, status, department, error, created_at, processed_at). Никаких сайд-кешей — GET /ingest/{job_id} читает ровно одну строку.
- **DI-скоуп**: `Scope.APP` — клиенты создаются один раз на процесс, teardown через `container.close()`.
- **Брокер — не в DI**: `src.ingestion.tasks.broker` — модульный singleton. Импортируется и API, и CLI-воркером; DI-биндинг порождал бы цикл API↔worker-контейнер.

## Переменные окружения (из `.env`)

`APP_HOST` / `APP_PORT` / `APP_ENV` / `LOG_LEVEL`
`API_KEYS` (csv) / `CORS_ORIGINS`
`RABBITMQ_URL=amqp://guest:guest@localhost:5672/`
`MILVUS_HOST` / `MILVUS_PORT=19530` / `MILVUS_COLLECTION=enterprise_kb`
`NEO4J_URI=bolt://localhost:7687` / `NEO4J_USER` / `NEO4J_PASSWORD`
`POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_HOST` / `POSTGRES_PORT`
`OLLAMA_HOST=http://localhost:11434` / `OLLAMA_MODEL=llama3.1:8b` / `EMBEDDING_MODEL=nomic-embed-text` / `EMBEDDING_DIM=1024`
`LIGHTRAG_WORKING_DIR=./data/lightrag` (+ внутри `lightrag_graph_storage` = `Neo4JStorage`)

## Известные расхождения с CLAUDE.md (aspirational → фактическое)

- Embedding: заявлен **BGE-M3** (dim=1024) → по-умолчанию `nomic-embed-text` (dim=768) в `.env`, хотя `EMBEDDING_DIM=1024` прописан — рассогласование, нужно привести к BGE-M3 когда Ollama pull-нется.
- LLM: заявлен **Llama 3.3 70B** → `OLLAMA_MODEL=llama3.1:8b` в `.env`.
- Reranker: заявлен **BGE-reranker-v2-m3** → `src.di.providers._stub_reranker` (cosine-порядок) подключён в DI; `_default_reranker()` с реальным cross-encoder живёт в `hybrid_search.py`, но не используется.
- Observability: заявлен **LangFuse** → только `@observe` декоратор (`_OBSERVE` fallback no-op), сервиса в `docker-compose.yml` нет.
- Eval: заявлен `python tests/eval/run_ragas.py` → папки `tests/eval/` не существует.
```
