"""Application settings — grouped per-backend via nested BaseSettings.

Each sub-class owns its own ``env_prefix`` and reads the same ``.env``
file.  Root :class:`Settings` composes them as fields, so consumers see
``settings.milvus.host``, ``settings.litellm.timeout_s`` etc.

Env-var naming
--------------
Every env var follows ``<PREFIX>_<FIELD>``, e.g. ``MILVUS_HOST``,
``POSTGRES_CONNECT_TIMEOUT_S``, ``LITELLM_EMBEDDING_MODEL``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = ".env"


def _sub(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=_ENV_FILE,
        extra="ignore",
    )


# ── sub-settings ─────────────────────────────────────────────────────


class ApiSettings(BaseSettings):
    model_config = _sub("API_")

    host: str = "0.0.0.0"
    port: int = 8000
    env: str = "development"
    log_level: str = "info"
    keys: str = "dev-key-change-me"
    cors_origins: str = "*"
    # Директория для загруженных файлов. Обязательно должна быть смонтирована
    # как shared volume между контейнерами api и worker в docker-compose —
    # worker читает тот же путь, что API записал.
    upload_dir: Path = Path("./data/uploads")

    @property
    def keys_set(self) -> set[str]:
        return {k.strip() for k in self.keys.split(",") if k.strip()}

    @property
    def cors_origins_list(self) -> list[str]:
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return origins or ["*"]


class MilvusSettings(BaseSettings):
    model_config = _sub("MILVUS_")

    host: str = "localhost"
    port: int = 19530
    collection: str = "enterprise_kb"
    timeout_s: float = 10.0
    # Размер выделенного ThreadPoolExecutor'а, в котором крутятся
    # блокирующие pymilvus-вызовы. Ограничиваем, чтобы при флапах Milvus
    # (когда `asyncio.wait_for` срабатывает, но сам thread ещё не
    # вернулся) не копились зомби-threads в default-executor'e.
    pool_size: int = 8


class Neo4jSettings(BaseSettings):
    model_config = _sub("NEO4J_")

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "changeme"
    timeout_s: float = 30.0


class PostgresSettings(BaseSettings):
    model_config = _sub("POSTGRES_")

    db: str = "enterprise_kb"
    user: str = "postgres"
    password: str = "postgres"
    host: str = "localhost"
    port: int = 5432
    connect_timeout_s: int = 10

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class LiteLLMSettings(BaseSettings):
    """LiteLLM proxy is OpenAI-API-compatible — клиенты ходят через
    ``openai.AsyncOpenAI(base_url=..., api_key=...)``.  Имена моделей —
    те же алиасы, что прописаны в LiteLLM ``config.yaml``.
    """

    model_config = _sub("LITELLM_")

    base_url: str = "http://localhost:4000"
    api_key: str = "sk-litellm-stub"
    model: str = "qwen2.5:3b"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    # HTTP-level timeout на каждый запрос в LiteLLM.  Должен быть **не
    # меньше** ``LIGHTRAG_LLM_TIMEOUT_S`` — иначе httpx оборвёт соединение
    # до того, как LightRAG сам решит таймаутить.
    timeout_s: float = 600.0


class RabbitMQSettings(BaseSettings):
    model_config = _sub("RABBITMQ_")

    url: str = "amqp://guest:guest@localhost:5672/"
    timeout_s: float = 10.0


class TaskiqSettings(BaseSettings):
    """Настройки для taskiq задач/воркера."""

    model_config = _sub("TASKIQ_")

    # Per-task timeout (секунды).  Стэмпается как label ``timeout`` на
    # ``process_document``; получатель оборачивает выполнение в
    # ``anyio.fail_after``.  Должен быть ≥ суммы LightRAG-таймаутов с
    # запасом (LLM + embed + Milvus + PG).
    task_timeout_s: int = 1800  # 30 мин
    # Сколько раз SimpleRetryMiddleware повторит упавшую задачу.
    # Учитывая длинные таймауты, 3 ретрая × 30 мин = 1.5 часа — много.
    max_retries: int = 2
    # Prefetch per-worker (QoS в aio-pika).  Сколько сообщений забирать
    # авансом.  Для медленных задач — низкое значение, чтобы не блокировать
    # другие воркеры.
    prefetch: int = 2


class LightRAGSettings(BaseSettings):
    """Knobs for LightRAG itself.

    Empty strings / ``0`` for model/dim mean "fall back to the shared
    :class:`LiteLLMSettings` values" — resolved via
    :meth:`Settings.effective_lightrag_llm_model` etc.
    """

    model_config = _sub("LIGHTRAG_")

    working_dir: str = "./data/lightrag"
    graph_storage: str = "Neo4JStorage"  # or "NetworkXStorage"
    llm_model: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0
    max_token_size: int = 8192
    # Per-вызов таймауты, прокидываются в LightRAG.__init__.  Дефолты
    # подняты относительно LightRAG-дефолтов (LLM 180s, embed 30s) —
    # у медленных локальных моделей генерация 7B/9B на CPU легко
    # превышает несколько минут на один запрос.
    llm_timeout_s: int = 600
    embedding_timeout_s: int = 120
    # Concurrency cap для LLM-вызовов.  LightRAG-дефолт=4; для слабых
    # локальных машин ставь 1-2, иначе параллельные запросы забивают
    # очередь у downstream-LLM (через LiteLLM).
    max_async: int = 2
    # Контекстное окно модели в токенах. Прокидывается в downstream-Ollama
    # через ``extra_body.options.num_ctx`` (LiteLLM пробрасывает в Ollama
    # как `options.num_ctx`).  Дефолт Ollama = 2048 режет entity-extraction
    # prompt LightRAG (chunk + system prompt ≈ 2000 ток. → "input length
    # exceeds context length"). Держи ≥ 8192; для больших документов
    # 16384-32768.  Для не-Ollama backends LiteLLM значение игнорирует.
    num_ctx: int = 16384
    # Параллельные эмбеддинг-вызовы. LightRAG-дефолт 8. На CPU
    # захлебнётся >4, на GPU смело 16-32.
    embedding_func_max_async: int = 8
    # Размер батча в одном embed-вызове. Дефолт 10; для больших
    # документов можно поднять до 32, если embed-модель поддерживает.
    embedding_batch_num: int = 10
    # Сколько документов обрабатывать одновременно (семафор вокруг
    # ainsert). LightRAG-дефолт 2. Подними до 4-8 если taskiq worker
    # запущен с --max-async-tasks 4+, иначе лишние tasks будут ждать.
    max_parallel_insert: int = 2
    # Сколько дополнительных проходов LLM делает на один чанк чтобы
    # добрать пропущенные сущности. 0 → только primary extract, 1 → +1
    # glean, 2 → +2. Каждый проход = отдельный LLM-вызов → растёт время.
    entity_extract_max_gleaning: int = 1


class IngestionSettings(BaseSettings):
    model_config = _sub("INGESTION_")

    batch_size: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 50
    # langchain-experimental SemanticChunker breakpoint strategy
    breakpoint_type: str = "percentile"
    breakpoint_amount: float | None = None
    # Document summary — LLM генерирует 2-3 предложения при ingestion.
    # Хранится в documents.summary (PG). Используется для hierarchical
    # two-stage search (Phase 2a).
    summary_enabled: bool = True
    summary_prompt: str = (
        "Summarize the following document in 2-3 sentences. "
        "Focus on the key topics and facts. Reply with just the summary."
    )


# ── root ─────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Composed application config.  Instantiated once as ``settings``."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    api: ApiSettings = Field(default_factory=ApiSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    litellm: LiteLLMSettings = Field(default_factory=LiteLLMSettings)
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)
    taskiq: TaskiqSettings = Field(default_factory=TaskiqSettings)
    lightrag: LightRAGSettings = Field(default_factory=LightRAGSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)

    # ── LightRAG effective values ────────────────────────────────────

    @property
    def effective_lightrag_llm_model(self) -> str:
        return self.lightrag.llm_model or self.litellm.model

    @property
    def effective_lightrag_embedding_model(self) -> str:
        return self.lightrag.embedding_model or self.litellm.embedding_model

    @property
    def effective_lightrag_embedding_dim(self) -> int:
        return self.lightrag.embedding_dim or self.litellm.embedding_dim

    @property
    def effective_llm_model(self) -> str:
        """Model name for direct LLM calls (judge, decomposer, summaries)."""
        return self.litellm.model


settings = Settings()
