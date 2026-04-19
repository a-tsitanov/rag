#!/usr/bin/env python3
"""CLI for database management: status, reset, recreate.

Usage:
    python scripts/db.py status          # check all connections
    python scripts/db.py reset           # drop + recreate everything
    python scripts/db.py reset milvus    # drop + recreate Milvus only
    python scripts/db.py reset neo4j     # clear Neo4j only
    python scripts/db.py reset postgres  # drop + recreate PG tables
    python scripts/db.py setup           # create if not exists (idempotent)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click

from src.config import settings

# ── output helpers ───────────────────────────────────────────────────

_COL = 28


def _step(label: str):
    dots = "." * max(1, _COL - len(label))
    click.echo(f"   {label} {dots} ", nl=False)


def _ok(detail: str = ""):
    click.secho("OK" + (f" ({detail})" if detail else ""), fg="green")


def _fail(err) -> bool:
    click.secho(f"FAIL — {err}", fg="red")
    return False


def _warn(msg: str):
    click.secho(msg, fg="yellow")


# ── Milvus ───────────────────────────────────────────────────────────

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200


def _milvus_uri() -> str:
    return f"http://{settings.milvus.host}:{settings.milvus.port}"


def _milvus_client():
    from pymilvus import MilvusClient
    return MilvusClient(uri=_milvus_uri())


def milvus_status() -> bool:
    _step("Milvus")
    try:
        client = _milvus_client()
        version = client.get_server_version().lstrip("v")
        has_col = client.has_collection(settings.milvus.collection)
        col_info = ""
        if has_col:
            stats = client.get_collection_stats(settings.milvus.collection)
            count = stats.get("row_count", "?")
            col_info = f", collection={settings.milvus.collection} rows={count}"
        client.close()
        _ok(f"v{version}{col_info}")
        return True
    except Exception as e:
        return _fail(e)


def milvus_drop() -> bool:
    _step(f"Drop {settings.milvus.collection}")
    try:
        client = _milvus_client()
        if client.has_collection(settings.milvus.collection):
            client.drop_collection(settings.milvus.collection)
            client.close()
            _ok("dropped")
        else:
            client.close()
            _ok("not found, skip")
        return True
    except Exception as e:
        return _fail(e)


def milvus_create() -> bool:
    collection = settings.milvus.collection
    _step(f"Create {collection}")
    try:
        from pymilvus import (
            CollectionSchema,
            DataType,
            FieldSchema,
            MilvusClient,
        )

        client = MilvusClient(uri=_milvus_uri())

        if client.has_collection(collection):
            _ok("already exists")
            client.close()
            return True

        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR,
                    is_primary=True, max_length=128,
                ),
                FieldSchema(
                    name="content", dtype=DataType.VARCHAR, max_length=65535,
                ),
                FieldSchema(
                    name="embedding", dtype=DataType.FLOAT_VECTOR,
                    dim=settings.effective_lightrag_embedding_dim,
                ),
                FieldSchema(
                    name="sparse_embedding",
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                ),
                FieldSchema(
                    name="doc_id", dtype=DataType.VARCHAR, max_length=128,
                ),
                FieldSchema(
                    name="department", dtype=DataType.VARCHAR, max_length=64,
                ),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(
                    name="doc_type", dtype=DataType.VARCHAR, max_length=64,
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
        _ok("created (8 fields, HNSW + sparse)")
        return True
    except Exception as e:
        return _fail(e)


# ── Neo4j ────────────────────────────────────────────────────────────

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


def _neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        settings.neo4j.uri,
        auth=(settings.neo4j.user, settings.neo4j.password),
    )


def neo4j_status() -> bool:
    _step("Neo4j")
    try:
        driver = _neo4j_driver()
        info = driver.get_server_info()
        with driver.session() as s:
            result = s.run("MATCH (n) RETURN count(n) AS cnt")
            cnt = result.single()["cnt"]
        driver.close()
        _ok(f"{info.agent}, {cnt} nodes")
        return True
    except Exception as e:
        return _fail(e)


def neo4j_clear() -> bool:
    _step("Clear all nodes/edges")
    try:
        driver = _neo4j_driver()
        with driver.session() as s:
            result = s.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
            deleted = result.single()["deleted"]
        driver.close()
        _ok(f"deleted {deleted} nodes")
        return True
    except Exception as e:
        return _fail(e)


def neo4j_setup_constraints() -> bool:
    from neo4j.exceptions import ClientError, DatabaseError

    driver = _neo4j_driver()
    ok = True

    for label, cypher in _NEO4J_CONSTRAINTS:
        _step(label)
        try:
            with driver.session() as s:
                s.run(cypher)
            _ok()
        except (ClientError, DatabaseError) as e:
            if "NODE KEY" in str(e) or "Enterprise" in str(e):
                try:
                    with driver.session() as s:
                        s.run(
                            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
                        )
                    _ok("fallback: name UNIQUE")
                except Exception as e2:
                    _fail(e2)
                    ok = False
            else:
                _fail(e)
                ok = False
        except Exception as e:
            _fail(e)
            ok = False

    for label, cypher in _NEO4J_INDEXES:
        _step(label)
        try:
            with driver.session() as s:
                s.run(cypher)
            _ok()
        except Exception as e:
            _fail(e)
            ok = False

    driver.close()
    return ok


# ── PostgreSQL ───────────────────────────────────────────────────────

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

_PG_INDEXES = [
    ("ix_documents_status", "CREATE INDEX IF NOT EXISTS ix_documents_status ON documents (status)"),
    ("ix_documents_department", "CREATE INDEX IF NOT EXISTS ix_documents_department ON documents (department)"),
    ("ix_chunks_doc_id", "CREATE INDEX IF NOT EXISTS ix_chunks_doc_id ON chunks (doc_id)"),
]


def _pg_conn():
    import psycopg
    return psycopg.connect(settings.postgres.dsn, autocommit=True)


def pg_status() -> bool:
    _step("PostgreSQL")
    try:
        conn = _pg_conn()
        row = conn.execute("SELECT current_database(), version()").fetchone()
        docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        conn.close()
        db_name = row[0]
        pg_ver = row[1].split(",")[0]
        _ok(f"{db_name} — {pg_ver}, {docs} documents")
        return True
    except Exception as e:
        return _fail(e)


def pg_drop() -> bool:
    _step("Drop tables")
    try:
        conn = _pg_conn()
        conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
        conn.execute("DROP TABLE IF EXISTS documents CASCADE")
        conn.close()
        _ok("dropped chunks, documents")
        return True
    except Exception as e:
        return _fail(e)


def pg_truncate() -> bool:
    _step("Truncate tables")
    try:
        conn = _pg_conn()
        conn.execute("TRUNCATE chunks, documents CASCADE")
        conn.close()
        _ok("truncated")
        return True
    except Exception as e:
        return _fail(e)


def pg_create() -> bool:
    conn = _pg_conn()
    ok = True

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
            ok = False

    for idx_name, ddl in _PG_INDEXES:
        _step(idx_name)
        try:
            conn.execute(ddl)
            _ok()
        except Exception as e:
            _fail(e)
            ok = False

    conn.close()
    return ok


# ── CLI ──────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Enterprise KB — database management CLI."""


@cli.command()
def status():
    """Check connectivity and show stats for all databases."""
    click.echo("\n  Database Status")
    click.echo("  " + "=" * 40)
    results = [pg_status(), milvus_status(), neo4j_status()]
    failed = results.count(False)
    if failed:
        click.echo(f"\n  {failed} service(s) unreachable.")
        click.echo("  Run: docker compose -f docker-compose.dev.yml up -d")
        raise SystemExit(1)
    click.echo()


@cli.command()
@click.argument("target", required=False, default="all",
                type=click.Choice(["all", "milvus", "neo4j", "postgres"]))
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
def reset(target: str, yes: bool):
    """Drop and recreate databases. Destroys all data."""
    targets = [target] if target != "all" else ["milvus", "neo4j", "postgres"]

    if not yes:
        names = ", ".join(targets)
        click.confirm(
            click.style(f"\n  This will DELETE all data in: {names}. Continue?", fg="yellow"),
            abort=True,
        )

    click.echo(f"\n  Resetting: {', '.join(targets)}")
    click.echo("  " + "=" * 40)

    if "milvus" in targets:
        milvus_drop()
        milvus_create()

    if "neo4j" in targets:
        neo4j_clear()
        neo4j_setup_constraints()

    if "postgres" in targets:
        pg_drop()
        pg_create()

    click.echo("\n  Done.\n")


@cli.command()
@click.argument("target", required=False, default="all",
                type=click.Choice(["all", "milvus", "neo4j", "postgres"]))
def setup(target: str):
    """Create schemas if not exist (idempotent). Same as setup_db.py."""
    targets = [target] if target != "all" else ["milvus", "neo4j", "postgres"]

    click.echo(f"\n  Setup: {', '.join(targets)}")
    click.echo("  " + "=" * 40)

    if "milvus" in targets:
        milvus_create()

    if "neo4j" in targets:
        neo4j_setup_constraints()

    if "postgres" in targets:
        pg_create()

    click.echo("\n  Done.\n")


@cli.command()
@click.argument("target", required=False, default="all",
                type=click.Choice(["all", "milvus", "neo4j", "postgres"]))
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
def clear(target: str, yes: bool):
    """Clear data but keep schemas (truncate tables, delete nodes, drop collection + recreate)."""
    targets = [target] if target != "all" else ["milvus", "neo4j", "postgres"]

    if not yes:
        names = ", ".join(targets)
        click.confirm(
            click.style(f"\n  This will clear all data in: {names}. Continue?", fg="yellow"),
            abort=True,
        )

    click.echo(f"\n  Clearing: {', '.join(targets)}")
    click.echo("  " + "=" * 40)

    if "milvus" in targets:
        milvus_drop()
        milvus_create()

    if "neo4j" in targets:
        neo4j_clear()

    if "postgres" in targets:
        pg_truncate()

    click.echo("\n  Done.\n")


if __name__ == "__main__":
    cli()
