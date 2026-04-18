#!/usr/bin/env python3
"""Setup all databases for Enterprise Knowledge Base.

Checks connectivity and creates schemas for Milvus, Neo4j, PostgreSQL.
Run after: docker compose up -d rabbitmq postgres milvus neo4j
"""

import sys
from pathlib import Path

# project root on sys.path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402

# ── output helpers ────────────────────────────────────────────────────

_COL = 26


def _step(label: str):
    dots = "." * max(1, _COL - len(label))
    print(f"   {label} {dots} ", end="", flush=True)


def _ok(detail: str = ""):
    print("OK" + (f" ({detail})" if detail else ""))


def _fail(err) -> bool:
    print(f"FAIL — {err}")
    return False


def _section(num: int, title: str):
    print(f"\n{num}. {title}")


# ── 1. Connectivity checks ───────────────────────────────────────────


def check_postgres() -> bool:
    _step("PostgreSQL")
    try:
        import psycopg

        with psycopg.connect(settings.postgres.dsn) as conn:
            row = conn.execute("SELECT current_database(), version()").fetchone()
        db_name = row[0]
        pg_ver = row[1].split(",")[0]  # "PostgreSQL 16.x ..."
        _ok(f"{db_name} — {pg_ver}")
        return True
    except Exception as e:
        return _fail(e)


def check_milvus() -> bool:
    _step("Milvus")
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(
            uri=f"http://{settings.milvus.host}:{settings.milvus.port}"
        )
        version = client.get_server_version().lstrip("v")
        client.close()
        _ok(f"v{version}")
        return True
    except Exception as e:
        return _fail(e)


def check_neo4j() -> bool:
    _step("Neo4j")
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password),
        )
        info = driver.get_server_info()
        driver.close()
        _ok(info.agent)
        return True
    except Exception as e:
        return _fail(e)


# ── 2. Milvus — collection ───────────────────────────────────────────

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200


def setup_milvus() -> bool:
    collection = settings.milvus.collection
    _step(collection)
    try:
        from pymilvus import (
            CollectionSchema,
            DataType,
            FieldSchema,
            MilvusClient,
        )

        client = MilvusClient(
            uri=f"http://{settings.milvus.host}:{settings.milvus.port}"
        )

        if client.has_collection(collection):
            _ok("already exists")
            client.close()
            return True

        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=128,
                ),
                FieldSchema(
                    name="content", dtype=DataType.VARCHAR, max_length=65535
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=settings.ollama.embedding_dim,
                ),
                FieldSchema(
                    name="sparse_embedding",
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                ),
                FieldSchema(
                    name="doc_id", dtype=DataType.VARCHAR, max_length=128
                ),
                FieldSchema(
                    name="department", dtype=DataType.VARCHAR, max_length=64
                ),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(
                    name="doc_type", dtype=DataType.VARCHAR, max_length=64
                ),
            ]
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
        )
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        client.create_collection(
            collection_name=collection,
            schema=schema,
            index_params=index_params,
        )
        client.close()
        _ok("created (8 fields, HNSW + sparse index)")
        return True
    except Exception as e:
        return _fail(e)


# ── 3. Neo4j — constraints & indexes ─────────────────────────────────

_NEO4J_CONSTRAINTS = [
    (
        "Entity (name,type) KEY",
        "CREATE CONSTRAINT entity_name_type IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE (e.name, e.type) IS NODE KEY",
    ),
    (
        "Document.id UNIQUE",
        "CREATE CONSTRAINT doc_id IF NOT EXISTS "
        "FOR (d:Document) REQUIRE d.id IS UNIQUE",
    ),
    (
        "Chunk.id UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
        "FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    ),
]

_NEO4J_INDEXES = [
    (
        "Index Entity.type",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    ),
    (
        "Index Document.department",
        "CREATE INDEX doc_dept IF NOT EXISTS FOR (d:Document) ON (d.department)",
    ),
    (
        "Index Document.status",
        "CREATE INDEX doc_status IF NOT EXISTS FOR (d:Document) ON (d.status)",
    ),
]


def setup_neo4j() -> bool:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ClientError, DatabaseError

    driver = GraphDatabase.driver(
        settings.neo4j.uri,
        auth=(settings.neo4j.user, settings.neo4j.password),
    )

    all_ok = True

    for label, cypher in _NEO4J_CONSTRAINTS:
        _step(label)
        try:
            with driver.session() as session:
                session.run(cypher)
            _ok()
        except (ClientError, DatabaseError) as e:
            if "NODE KEY" in str(e) or "Enterprise" in str(e):
                try:
                    with driver.session() as session:
                        session.run(
                            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
                        )
                    _ok("fallback: name UNIQUE")
                except Exception as e2:
                    _fail(e2)
                    all_ok = False
            else:
                _fail(e)
                all_ok = False
        except Exception as e:
            _fail(e)
            all_ok = False

    for label, cypher in _NEO4J_INDEXES:
        _step(label)
        try:
            with driver.session() as session:
                session.run(cypher)
            _ok()
        except Exception as e:
            _fail(e)
            all_ok = False

    driver.close()
    return all_ok


# ── 4. PostgreSQL — tables & indexes ─────────────────────────────────

_PG_TABLES = [
    (
        "documents",
        """
        CREATE TABLE IF NOT EXISTS documents (
            id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
            path          TEXT         NOT NULL,
            status        VARCHAR(32)  NOT NULL DEFAULT 'pending',
            department    VARCHAR(64),
            error         TEXT,
            summary       TEXT,
            created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
            processed_at  TIMESTAMPTZ
        )
        """,
    ),
    (
        "chunks",
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id              UUID     PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id          UUID     NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            content         TEXT     NOT NULL,
            position        INTEGER  NOT NULL,
            chunk_metadata  JSONB    NOT NULL DEFAULT '{}'::jsonb
        )
        """,
    ),
]

_PG_MIGRATIONS = [
    (
        "documents.error column",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS error TEXT",
    ),
    (
        "documents.summary column",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary TEXT",
    ),
]

_PG_INDEXES = [
    (
        "ix_documents_status",
        "CREATE INDEX IF NOT EXISTS ix_documents_status ON documents (status)",
    ),
    (
        "ix_documents_department",
        "CREATE INDEX IF NOT EXISTS ix_documents_department ON documents (department)",
    ),
    (
        "ix_chunks_doc_id",
        "CREATE INDEX IF NOT EXISTS ix_chunks_doc_id ON chunks (doc_id)",
    ),
]


def setup_postgres() -> bool:
    import psycopg

    conn = psycopg.connect(settings.postgres.dsn, autocommit=True)
    all_ok = True

    for table_name, ddl in _PG_TABLES:
        _step(table_name)
        try:
            conn.execute(ddl)
            row = conn.execute(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s",
                (table_name,),
            ).fetchone()
            _ok(f"{row[0]} columns")
        except Exception as e:
            _fail(e)
            all_ok = False

    for label, ddl in _PG_MIGRATIONS:
        _step(label)
        try:
            conn.execute(ddl)
            _ok()
        except Exception as e:
            _fail(e)
            all_ok = False

    for idx_name, ddl in _PG_INDEXES:
        _step(idx_name)
        try:
            conn.execute(ddl)
            _ok()
        except Exception as e:
            _fail(e)
            all_ok = False

    conn.close()
    return all_ok


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    print("\nEnterprise KB — Database Setup")
    print("=" * 34)

    # 1. connectivity
    _section(1, "Checking connections")
    services = {
        "PostgreSQL": check_postgres,
        "Milvus": check_milvus,
        "Neo4j": check_neo4j,
    }
    results = {name: fn() for name, fn in services.items()}
    failed = [n for n, ok in results.items() if not ok]

    if failed:
        print(f"\n   Cannot reach: {', '.join(failed)}")
        print("   Start services: docker compose up -d rabbitmq postgres milvus neo4j")
        return 1

    # 2. Milvus
    _section(2, "Milvus — collection")
    setup_milvus()

    # 3. Neo4j
    _section(3, "Neo4j — constraints & indexes")
    setup_neo4j()

    # 4. PostgreSQL
    _section(4, "PostgreSQL — tables & indexes")
    setup_postgres()

    print("\nDone. All services initialized.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
