from pymilvus import MilvusClient

from src.config import settings


class MilvusService:
    def __init__(self):
        self.client: MilvusClient | None = None

    async def connect(self):
        self.client = MilvusClient(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
        )
        self._ensure_collection()

    async def disconnect(self):
        if self.client:
            self.client.close()

    def _ensure_collection(self):
        if not self.client.has_collection(settings.milvus_collection):
            from pymilvus import CollectionSchema, DataType, FieldSchema

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields=fields)
            self.client.create_collection(
                collection_name=settings.milvus_collection,
                schema=schema,
            )

    async def insert(self, data: list[dict]):
        self.client.insert(
            collection_name=settings.milvus_collection,
            data=data,
        )

    async def search(
        self, vector: list[float], top_k: int = 5
    ) -> list[dict]:
        results = self.client.search(
            collection_name=settings.milvus_collection,
            data=[vector],
            limit=top_k,
            output_fields=["text", "document_id", "chunk_index"],
        )
        return results[0] if results else []


milvus_service = MilvusService()
