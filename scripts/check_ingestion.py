"""Diagnostic: show what ended up in each storage backend after ingestion.

Checks:
  1. Redis   — job:* hashes (status per job)
  2. Postgres — documents table (rows, status, summary)
  3. Milvus  — chunk count per doc_id
  4. Neo4j   — Entity node + HAS_ENTITY relation counts
  5. LightRAG working dir — KV / vector-DB JSON file sizes

Run::

    python -m scripts.check_ingestion
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402


_SEP = "─" * 70


async def check_redis() -> None:
    import redis.asyncio as aioredis

    print(_SEP)
    print("Redis — ingestion jobs")
    print(_SEP)

    # Reuse REDIS_URL from env if set, else fall back to localhost.
    import os
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    r = aioredis.from_url(redis_url, decode_responses=False)
    try:
        await r.ping()
    except Exception as exc:
        print(f"  redis unreachable: {exc}\n")
        return

    keys = []
    cursor = 0
    while True:
        cursor, chunk = await r.scan(cursor, match="job:*", count=200)
        keys.extend(chunk)
        if cursor == 0:
            break

    if not keys:
        print("  (no job:* hashes — nothing enqueued since last flush)\n")
        await r.aclose()
        return

    print(f"  {len(keys)} jobs total\n")
    print(f"  {'status':<12} {'department':<14} {'attempts':<8}  path")
    for key in keys[:20]:
        data = await r.hgetall(key)
        d = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in data.items()
        }
        print(
            f"  {d.get('status', '?'):<12} "
            f"{(d.get('department') or '—'):<14} "
            f"{d.get('attempts', '0'):<8}  "
            f"{d.get('path', '—')}",
        )
    if len(keys) > 20:
        print(f"  … +{len(keys) - 20} more")
    print()
    await r.aclose()


async def check_postgres() -> None:
    import psycopg

    print(_SEP)
    print("PostgreSQL — `documents` + `chunks` tables")
    print(_SEP)

    try:
        conn = await psycopg.AsyncConnection.connect(
            settings.postgres.dsn, autocommit=True,
        )
    except Exception as exc:
        print(f"  postgres unreachable: {exc}\n")
        return

    try:
        total = await (await conn.execute(
            "SELECT count(*) FROM documents",
        )).fetchone()
        by_status = await (await conn.execute(
            "SELECT status, count(*) FROM documents GROUP BY status "
            "ORDER BY count DESC",
        )).fetchall()

        print(f"  documents total: {total[0]}")
        for status, n in by_status:
            print(f"    {status:<12} {n}")
        print()

        # last 10 documents
        rows = await (await conn.execute(
            "SELECT id, status, department, "
            "COALESCE(path, ''), COALESCE(summary, ''), processed_at "
            "FROM documents "
            "ORDER BY processed_at DESC NULLS LAST LIMIT 10",
        )).fetchall()

        if rows:
            print("  last 10 rows:")
            for i, (did, status, dept, path, summary, processed) in enumerate(rows, 1):
                short_id = str(did)[:8]
                short_path = (path or "—").rsplit("/", 1)[-1][:40]
                short_sum = (summary or "").replace("\n", " ")[:70]
                print(
                    f"    {i:>2}. {short_id}  {status:<10} "
                    f"{(dept or '—'):<10} {short_path:<40}",
                )
                if short_sum:
                    print(f"        summary: {short_sum!r}")

        # chunks table (if worker writes there)
        try:
            c = await (await conn.execute("SELECT count(*) FROM chunks")).fetchone()
            print(f"\n  chunks total: {c[0]}")
        except Exception:
            print("\n  chunks: table empty or missing (chunks live in Milvus)")
        print()
    finally:
        await conn.close()


async def check_milvus() -> None:
    from pymilvus import MilvusClient

    print(_SEP)
    print(f"Milvus — collection `{settings.milvus.collection}`")
    print(_SEP)

    try:
        client = MilvusClient(
            uri=f"http://{settings.milvus.host}:{settings.milvus.port}",
        )
        if not client.has_collection(settings.milvus.collection):
            print(f"  collection missing\n")
            return

        stats = client.get_collection_stats(settings.milvus.collection)
        print(f"  stats: {stats}")

        # chunks per doc_id
        res = client.query(
            collection_name=settings.milvus.collection,
            filter="",
            output_fields=["doc_id", "department", "doc_type"],
            limit=10_000,
        )
        from collections import Counter
        by_doc = Counter(r["doc_id"] for r in res)
        by_dept = Counter(r.get("department", "") or "—" for r in res)

        print(f"  rows sampled: {len(res)}")
        print(f"  unique doc_ids: {len(by_doc)}")
        print("  top 10 by doc_id:")
        for doc_id, n in by_doc.most_common(10):
            print(f"    {doc_id[:36]:<36}  {n} chunks")
        print("\n  chunks per department:")
        for d, n in by_dept.most_common():
            print(f"    {d:<14} {n}")
        print()
    except Exception as exc:
        print(f"  milvus error: {exc}\n")


async def check_neo4j() -> None:
    from neo4j import AsyncGraphDatabase

    print(_SEP)
    print("Neo4j — LightRAG knowledge graph")
    print(_SEP)

    try:
        driver = AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password),
        )
        async with driver.session() as s:
            # total nodes + edges
            row = await (await s.run(
                "MATCH (n) RETURN labels(n) AS label, count(*) AS cnt "
                "ORDER BY cnt DESC",
            )).data()
            print(f"  nodes by label:")
            for r in row[:10]:
                print(f"    {str(r['label']):<30} {r['cnt']}")

            edges = await (await s.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt "
                "ORDER BY cnt DESC LIMIT 10",
            )).data()
            print("\n  edges by type:")
            for r in edges:
                print(f"    {r['rel']:<30} {r['cnt']}")

            # top entity names
            ents = await (await s.run(
                "MATCH (n) WHERE 'entity_id' IN keys(properties(n)) OR "
                "n.name IS NOT NULL "
                "RETURN coalesce(n.entity_id, n.name) AS name, "
                "       n.entity_type AS type "
                "LIMIT 15",
            )).data()
            if ents:
                print("\n  sample entities:")
                for r in ents:
                    print(f"    {str(r['name'])[:50]:<50} ({r.get('type') or '—'})")
            print()

        await driver.close()
    except Exception as exc:
        print(f"  neo4j error: {exc}\n")


def check_lightrag_dir() -> None:
    print(_SEP)
    print(f"LightRAG working dir — {settings.lightrag.working_dir}")
    print(_SEP)

    wd = Path(settings.lightrag.working_dir)
    if not wd.exists():
        print(f"  dir missing — no lightrag data yet\n")
        return

    files = sorted(wd.glob("*.json"))
    if not files:
        print(f"  {wd}: empty\n")
        return

    for f in files:
        size = f.stat().st_size
        n_items = "?"
        try:
            if size < 5_000_000:
                data = json.loads(f.read_text())
                if isinstance(data, dict):
                    n_items = str(len(data))
                elif isinstance(data, list):
                    n_items = str(len(data))
        except Exception:
            pass
        print(f"  {f.name:<40} {size:>10,} bytes  ({n_items} items)")
    print()


async def main() -> None:
    await check_redis()
    await check_postgres()
    await check_milvus()
    await check_neo4j()
    check_lightrag_dir()


if __name__ == "__main__":
    asyncio.run(main())
