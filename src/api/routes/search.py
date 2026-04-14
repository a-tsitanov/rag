"""POST /api/v1/search — hybrid RAG + vector search."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import require_api_key
from src.models.search import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_api_key)],
    summary="Hybrid semantic search",
    description=(
        "Runs a hybrid retrieval pipeline:\n\n"
        "1. Embeds the query and fetches candidates from Milvus "
        "(optionally filtered by ``department``).\n"
        "2. Reranks candidates with BGE-reranker-v2-m3.\n"
        "3. Calls LightRAG to generate a natural-language answer in the "
        "requested ``mode``.\n\n"
        "Returns the generated ``answer``, reranked ``sources`` with "
        "``doc_id`` / ``position``, and total ``latency_ms``."
    ),
    responses={
        401: {"description": "Missing X-API-Key"},
        403: {"description": "Invalid X-API-Key"},
        503: {"description": "Search subsystem unavailable"},
    },
)
async def search(req: SearchRequest, request: Request) -> SearchResponse:
    searcher = getattr(request.app.state, "searcher", None)
    if searcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search subsystem is not initialised "
            "(check /health for backend status).",
        )

    try:
        return await searcher.search(
            query=req.query,
            mode=req.mode,  # type: ignore[arg-type]
            department=req.department,
            top_k=req.top_k,
            user_id=req.user_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {exc}",
        ) from exc
