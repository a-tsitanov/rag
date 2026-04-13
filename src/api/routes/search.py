from fastapi import APIRouter, Depends

from src.api.dependencies import get_milvus
from src.models.search import SearchRequest, SearchResponse, SearchResult
from src.services.milvus import MilvusService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search_knowledge_base(
    request: SearchRequest,
    milvus: MilvusService = Depends(get_milvus),
):
    # TODO: implement full RAG search pipeline
    # 1. Embed query
    # 2. Vector search in Milvus
    # 3. Graph traversal in Neo4j for context enrichment
    # 4. LightRAG for answer generation
    return SearchResponse(query=request.query, results=[], answer="")
