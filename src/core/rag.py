from lightrag import LightRAG, QueryParam

from src.config import settings


class RAGEngine:
    def __init__(self):
        self.rag: LightRAG | None = None

    async def initialize(self):
        self.rag = LightRAG(working_dir=settings.lightrag_working_dir)

    async def insert(self, text: str):
        if self.rag is None:
            await self.initialize()
        await self.rag.ainsert(text)

    async def query(self, question: str, mode: str = "hybrid") -> str:
        if self.rag is None:
            await self.initialize()
        return await self.rag.aquery(
            question,
            param=QueryParam(mode=mode),
        )


rag_engine = RAGEngine()
