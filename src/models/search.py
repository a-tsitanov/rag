from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict = {}


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    answer: str
