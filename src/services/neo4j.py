from neo4j import GraphDatabase

from src.config import settings


class Neo4jService:
    def __init__(self):
        self.driver = None

    def connect(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def disconnect(self):
        if self.driver:
            self.driver.close()

    def create_document_node(self, doc_id: str, metadata: dict):
        with self.driver.session() as session:
            session.run(
                "MERGE (d:Document {id: $id}) SET d += $props",
                id=doc_id,
                props=metadata,
            )

    def create_chunk_node(self, chunk_id: str, doc_id: str, text: str):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text,
            )

    def create_entity_relation(self, entity1: str, relation: str, entity2: str):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (e1:Entity {name: $e1})
                MERGE (e2:Entity {name: $e2})
                MERGE (e1)-[r:RELATES {type: $rel}]->(e2)
                """,
                e1=entity1,
                e2=entity2,
                rel=relation,
            )

    def get_related_entities(self, entity: str, depth: int = 2) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $name})-[r*1..$depth]-(related)
                RETURN related.name AS name, labels(related) AS labels
                LIMIT 20
                """,
                name=entity,
                depth=depth,
            )
            return [record.data() for record in result]


neo4j_service = Neo4jService()
