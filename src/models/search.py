from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"           # naive | local | global | hybrid
    department: str | None = None
    top_k: int = 10
    user_id: str | None = None


class SourceCitation(BaseModel):
    doc_id: str
    chunk_id: str
    position: int
    content: str
    score: float
    department: str = ""
    doc_type: str = ""


class SearchResponse(BaseModel):
    query: str
    answer: str
    mode: str
    sources: list[SourceCitation] = Field(default_factory=list)
    latency_ms: float = 0.0
