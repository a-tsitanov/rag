import pytest
import pytest_asyncio

from src.storage.neo4j_client import AsyncNeo4jClient, Entity

_NEO4J_PASSWORD = "password"


@pytest.fixture(scope="module")
def neo4j_container():
    """Start a Neo4j container once per module (sync, no event-loop issues)."""
    pytest.importorskip("testcontainers")
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    try:
        container = (
            DockerContainer("neo4j:5-community")
            .with_exposed_ports(7474, 7687)
            .with_env("NEO4J_AUTH", f"neo4j/{_NEO4J_PASSWORD}")
            .with_env("NEO4J_PLUGINS", '["apoc"]')
        )
        container.start()
        wait_for_logs(container, "Started.", timeout=90)
    except Exception as exc:
        pytest.skip(f"Neo4j container unavailable: {exc}")

    yield container
    container.stop()


@pytest_asyncio.fixture
async def neo4j(neo4j_container):
    """Per-test async client — each test gets its own driver bound to the
    current event loop, avoiding cross-loop errors."""
    host = neo4j_container.get_container_host_ip()
    port = neo4j_container.get_exposed_port(7687)
    uri = f"bolt://{host}:{port}"

    client = AsyncNeo4jClient(uri=uri, user="neo4j", password=_NEO4J_PASSWORD)
    await client.connect()
    yield client
    await client.disconnect()


# ── core scenario: 3 entities, 2 relations, get_neighbors ────────────


@pytest.mark.asyncio
async def test_entities_relations_and_neighbors(neo4j):
    # 3 entities
    await neo4j.upsert_entity("Python", "Language", {"version": "3.12"})
    await neo4j.upsert_entity("Django", "Framework", {"version": "5.0"})
    await neo4j.upsert_entity("FastAPI", "Framework", {"version": "0.115"})

    # 2 relations
    await neo4j.upsert_relation(
        "Python", "Django", "HAS_FRAMEWORK", {"since": "2005"},
    )
    await neo4j.upsert_relation(
        "Python", "FastAPI", "HAS_FRAMEWORK", {"since": "2018"},
    )

    # neighbours of Python at depth 1
    neighbors = await neo4j.get_neighbors("Python", depth=1)

    assert len(neighbors) == 2
    names = {e.name for e in neighbors}
    assert names == {"Django", "FastAPI"}

    for entity in neighbors:
        assert isinstance(entity, Entity)
        assert entity.type == "Framework"
        assert "version" in entity.properties


# ── upsert updates properties ────────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_entity_updates_properties(neo4j):
    await neo4j.upsert_entity("Redis", "Database", {"version": "7.0"})
    await neo4j.upsert_entity("Redis", "Database", {"version": "7.4", "mode": "cluster"})

    # reach Redis through a dummy relation
    await neo4j.upsert_entity("App", "Service", {})
    await neo4j.upsert_relation("App", "Redis", "USES", {})

    neighbors = await neo4j.get_neighbors("App", depth=1)
    redis = next(e for e in neighbors if e.name == "Redis")

    assert redis.properties["version"] == "7.4"
    assert redis.properties["mode"] == "cluster"


# ── depth traversal ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_neighbors_respects_depth(neo4j):
    # chain: X -> Y -> Z  (unique names to avoid collision with other tests)
    await neo4j.upsert_entity("X", "Node", {})
    await neo4j.upsert_entity("Y", "Node", {})
    await neo4j.upsert_entity("Z", "Node", {})

    await neo4j.upsert_relation("X", "Y", "LINKS_TO", {})
    await neo4j.upsert_relation("Y", "Z", "LINKS_TO", {})

    # depth=1 — only Y
    depth1 = await neo4j.get_neighbors("X", depth=1)
    assert {e.name for e in depth1} == {"Y"}

    # depth=2 — Y and Z
    depth2 = await neo4j.get_neighbors("X", depth=2)
    assert {e.name for e in depth2} == {"Y", "Z"}
