import logging
from dataclasses import dataclass, field

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import (
    ClientError,
    DatabaseError,
    ServiceUnavailable,
    SessionExpired,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

logger = logging.getLogger(__name__)

_retry_neo4j = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ServiceUnavailable, SessionExpired, ConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


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
    ):
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._driver = None

    async def connect(self):
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
            max_connection_pool_size=50,
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

    @_retry_neo4j
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

    @_retry_neo4j
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

    @_retry_neo4j
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
