"""Initialize Milvus collection and Neo4j constraints."""

from pymilvus import MilvusClient
from neo4j import GraphDatabase

from src.config import settings


def init_milvus():
    client = MilvusClient(
        uri=f"http://{settings.milvus.host}:{settings.milvus.port}"
    )
    if client.has_collection(settings.milvus.collection):
        print(f"Milvus collection '{settings.milvus.collection}' already exists")
    else:
        print("Milvus collection will be created on first insert")
    client.close()


def init_neo4j():
    driver = GraphDatabase.driver(
        settings.neo4j.uri,
        auth=(settings.neo4j.user, settings.neo4j.password),
    )
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        print("Neo4j constraints created")
    driver.close()


if __name__ == "__main__":
    print("Initializing databases...")
    init_milvus()
    init_neo4j()
    print("Done")
