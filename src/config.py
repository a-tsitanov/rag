from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    log_level: str = "info"

    # API auth — comma-separated X-API-Keys
    api_keys: str = "dev-key-change-me"
    cors_origins: str = "*"  # comma-separated; "*" = allow all

    @property
    def api_keys_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origins_list(self) -> list[str]:
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return origins or ["*"]

    # RabbitMQ (task broker)
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "enterprise_kb"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"

    # PostgreSQL
    postgres_db: str = "enterprise_kb"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 1024

    # OpenAI (optional fallback)
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # LightRAG
    lightrag_working_dir: str = "./data/lightrag"
    lightrag_graph_storage: str = "Neo4JStorage"  # or "NetworkXStorage"

    # Ingestion
    ingestion_batch_size: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 50


settings = Settings()
