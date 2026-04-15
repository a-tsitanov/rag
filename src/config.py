"""Application settings — grouped per-backend via nested BaseSettings.

Each sub-class owns its own ``env_prefix`` and reads the same ``.env``
file.  Root :class:`Settings` composes them as fields, so consumers see
``settings.milvus.host``, ``settings.ollama.timeout_s`` etc.

Env-var naming
--------------
Every env var follows ``<PREFIX>_<FIELD>``, e.g. ``MILVUS_HOST``,
``POSTGRES_CONNECT_TIMEOUT_S``, ``OLLAMA_EMBEDDING_MODEL``.
"""

from __future__ import annotations

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


class OllamaSettings(BaseSettings):
    model_config = _sub("OLLAMA_")

    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 1024
    timeout_s: float = 60.0


class OpenAISettings(BaseSettings):
    model_config = _sub("OPENAI_")

    api_key: str = ""
    llm_model: str = "gpt-4o-mini"


class RabbitMQSettings(BaseSettings):
    model_config = _sub("RABBITMQ_")

    url: str = "amqp://guest:guest@localhost:5672/"
    timeout_s: float = 10.0


class LightRAGSettings(BaseSettings):
    """Knobs for LightRAG itself.

    Empty strings / ``0`` for model/dim mean "fall back to the shared
    OllamaSettings values" — resolved via
    :meth:`Settings.effective_lightrag_llm_model` etc.
    """

    model_config = _sub("LIGHTRAG_")

    working_dir: str = "./data/lightrag"
    graph_storage: str = "Neo4JStorage"  # or "NetworkXStorage"
    llm_model: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0
    max_token_size: int = 8192


class IngestionSettings(BaseSettings):
    model_config = _sub("INGESTION_")

    batch_size: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 50


# ── root ─────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Composed application config.  Instantiated once as ``settings``."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    api: ApiSettings = Field(default_factory=ApiSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)
    lightrag: LightRAGSettings = Field(default_factory=LightRAGSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)

    # ── LightRAG effective values (fallback to Ollama shared config) ──

    @property
    def effective_lightrag_llm_model(self) -> str:
        return self.lightrag.llm_model or self.ollama.model

    @property
    def effective_lightrag_embedding_model(self) -> str:
        return self.lightrag.embedding_model or self.ollama.embedding_model

    @property
    def effective_lightrag_embedding_dim(self) -> int:
        return self.lightrag.embedding_dim or self.ollama.embedding_dim


settings = Settings()
