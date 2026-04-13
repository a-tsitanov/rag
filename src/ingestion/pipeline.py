import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.core.embeddings import embedding_service
from src.core.rag import rag_engine
from src.services.milvus import milvus_service
from src.services.neo4j import neo4j_service


class IngestionPipeline:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    async def process(self, document_id: str, filename: str, content: str):
        # 1. Split into chunks
        chunks = self.splitter.split_text(content)

        # 2. Generate embeddings
        embeddings = await embedding_service.embed_batch(
            [chunk for chunk in chunks]
        )

        # 3. Store in Milvus
        milvus_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_{i}"
            milvus_data.append({
                "id": chunk_id,
                "embedding": embedding,
                "text": chunk,
                "document_id": document_id,
                "chunk_index": i,
            })
        await milvus_service.insert(milvus_data)

        # 4. Store graph relationships in Neo4j
        neo4j_service.create_document_node(
            document_id, {"filename": filename, "chunk_count": len(chunks)}
        )
        for i, chunk in enumerate(chunks):
            neo4j_service.create_chunk_node(
                f"{document_id}_{i}", document_id, chunk[:500]
            )

        # 5. Index in LightRAG
        await rag_engine.insert(content)

        return {"document_id": document_id, "chunks": len(chunks)}


ingestion_pipeline = IngestionPipeline()
