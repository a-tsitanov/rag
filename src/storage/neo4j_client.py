from dataclasses import dataclass, field

from loguru import logger
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ClientError, DatabaseError

from src.config import settings

# Retry не делаем на уровне клиента — делает taskiq на уровне всей
# ingestion-задачи. Это уберёт двойной ретрай (3× tenacity × 2× taskiq).


@dataclass
class Entity:
    name: str
    type: str
    properties: dict = field(default_factory=dict)


class AsyncNeo4jClient:
    """Async Neo4j client with connection pooling, retry, and parameterised Cypher.

    All data values are passed as Cypher parameters — never interpolated.
    Relationship types use ``apoc.merge.relationship`` so the type string
    is also a parameter.  Variable-length traversal uses
    ``apoc.path.subgraphNodes`` so depth is a parameter too.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        *,
        connection_timeout: float | None = None,
    ):
        self._uri = uri or settings.neo4j.uri
        self._user = user or settings.neo4j.user
        self._password = password or settings.neo4j.password
        # 0 → без таймаута (драйвер воспримет как inf)
        self._connection_timeout = (
            connection_timeout
            if connection_timeout is not None
            else settings.neo4j.timeout_s
        )
        self._driver = None

    async def connect(self):
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
            max_connection_pool_size=50,
            connection_timeout=self._connection_timeout,
            max_transaction_retry_time=self._connection_timeout,
        )
        await self._ensure_constraints()

    async def disconnect(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ── schema ────────────────────────────────────────────────────────

    async def _ensure_constraints(self):
        try:
            async with self._driver.session() as session:
                # Enterprise: composite node key
                await session.run(
                    "CREATE CONSTRAINT entity_name_type IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE (e.name, e.type) IS NODE KEY"
                )
        except (ClientError, DatabaseError):
            # Community: fall back to name-only uniqueness
            logger.warning(
                "NODE KEY constraint unavailable (requires Enterprise), "
                "falling back to name-only uniqueness"
            )
            async with self._driver.session() as session:
                await session.run(
                    "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
                )

    # ── writes ────────────────────────────────────────────────────────

    async def upsert_entity(
        self,
        name: str,
        type: str,
        properties: dict | None = None,
    ) -> None:
        props = {
            k: v
            for k, v in (properties or {}).items()
            if k not in ("name", "type")
        }
        async with self._driver.session() as session:
            await session.run(
                "MERGE (e:Entity {name: $name, type: $type}) "
                "SET e += $props",
                name=name,
                type=type,
                props=props,
            )

    async def upsert_relation(
        self,
        from_entity: str,
        to_entity: str,
        rel_type: str,
        properties: dict | None = None,
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                "MATCH (a:Entity {name: $from_name}) "
                "MATCH (b:Entity {name: $to_name}) "
                "CALL apoc.merge.relationship(a, $rel_type, {}, {}, b) "
                "YIELD rel "
                "SET rel += $props "
                "RETURN rel",
                from_name=from_entity,
                to_name=to_entity,
                rel_type=rel_type,
                props=properties or {},
            )

    # ── reads ─────────────────────────────────────────────────────────

    async def get_neighbors(
        self,
        entity_name: str,
        depth: int = 2,
    ) -> list[Entity]:
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (start:Entity {name: $name}) "
                "CALL apoc.path.subgraphNodes(start, "
                "  {maxLevel: $depth, labelFilter: 'Entity'}) "
                "YIELD node "
                "WHERE node <> start "
                "RETURN DISTINCT "
                "  node.name AS name, "
                "  node.type AS type, "
                "  properties(node) AS props",
                name=entity_name,
                depth=depth,
            )
            return [
                Entity(
                    name=record["name"],
                    type=record["type"],
                    properties={
                        k: v
                        for k, v in record["props"].items()
                        if k not in ("name", "type")
                    },
                )
                async for record in result
            ]
