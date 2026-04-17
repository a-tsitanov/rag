"""POST /api/v1/search — hybrid RAG + vector search."""

from __future__ import annotations

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.api.auth import require_api_key
from src.models.search import SearchRequest, SearchResponse
from src.retrieval.hybrid_search import HybridSearcher

router = APIRouter(tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_api_key)],
    summary="Hybrid semantic search",
    description=(
        "Runs a hybrid retrieval pipeline:\n\n"
        "1. Embeds the query and fetches candidates from Milvus "
        "(with metadata filters: department, doc_type, date range).\n"
        "2. Reranks candidates.\n"
        "3. Calls LightRAG to generate a natural-language answer with "
        "configurable QueryParam knobs.\n\n"
        "Returns ``answer``, ``sources``, and ``latency_ms``."
    ),
    responses={
        401: {"description": "Missing X-API-Key"},
        403: {"description": "Invalid X-API-Key"},
        500: {"description": "Downstream failure"},
    },
)
@inject
async def search(
    req: SearchRequest,
    searcher: FromDishka[HybridSearcher],
) -> SearchResponse:
    try:
        return await searcher.search(
            query=req.query,
            mode=req.mode,  # type: ignore[arg-type]
            department=req.department,
            top_k=req.top_k,
            user_id=req.user_id,
            # Phase 1a: metadata filters
            doc_type_filter=req.doc_type_filter,
            created_after=req.created_after,
            created_before=req.created_before,
            # Phase 1b: LightRAG QueryParam knobs
            chunk_top_k=req.chunk_top_k,
            max_entity_tokens=req.max_entity_tokens,
            max_relation_tokens=req.max_relation_tokens,
            max_total_tokens=req.max_total_tokens,
            response_type=req.response_type,
            include_references=req.include_references,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {exc}",
        ) from exc
