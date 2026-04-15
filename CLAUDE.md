# Enterprise Knowledge Base — контекст проекта

## Архитектура
- **RAG**: LightRAG (НЕ GraphRAG — инкрементальные обновления)
- **Vector DB**: Milvus Distributed (порт 19530)
- **Graph DB**: Neo4j Enterprise (порт 7687)
- **Embedding**: BGE-M3 via ollama (порт 11434)
- **LLM**: Llama 3.3 70B via ollama / vLLM (порт 11434)
- **Queue**: RabbitMQ + taskiq (AMQP 5672, dashboard 15672)
- **API**: FastAPI + uvicorn (порт 8000)
- **Observability**: LangFuse self-hosted (порт 3000)

## Правила кода
- Python 3.11+, async/await везде где возможно
- Type hints обязательны для всех функций
- Pydantic v2 для валидации данных
- Логировать через structlog (JSON формат)
- Все DB операции через retry decorator (tenacity)
- Батчевая обработка: размер батча 50 документов

## Ключевые архитектурные решения
- LightRAG mode="hybrid" для всех запросов
- Семантический чанкинг (НЕ фиксированный размер)
- BGE-reranker-v2-m3 после retrieval
- Access control на уровне Milvus коллекций (по отделам)
- Инкрементальный update графа (НИКОГДА не перестраивать полностью)

## Запуск сервисов
```bash
docker compose up -d           # поднять инфру
python scripts/setup_db.py     # инициализировать БД
uvicorn src.api.main:app --reload  # API
```

## Тесты
```bash
pytest tests/unit/ -v          # быстрые тесты
pytest tests/integration/ -v   # требуют запущенных БД
python tests/eval/run_ragas.py # оценка качества RAG
```

## Переменные окружения (.env)
MILVUS_HOST, MILVUS_PORT=19530
NEO4J_URI, NEO4J_PASSWORD
OLLAMA_HOST=http://localhost:11434
LANGFUSE_HOST=http://localhost:3000
RABBITMQ_URL=amqp://guest:guest@localhost:5672/

## Целевые метрики
- Ingestion: 5000 документов/день
- Search latency: <500ms для hybrid
- Faithfulness RAGAS: >0.85
- Uptime API: 99.5%

## Статус задач (обновляй)
- [ ] Фаза 1: инфраструктура (Docker + DB clients)
- [ ] Фаза 2: ingestion pipeline
- [ ] Фаза 3: retrieval + LightRAG + API
- [ ] Фаза 4: observability + production