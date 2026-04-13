from pydantic import BaseModel


class IngestResponse(BaseModel):
    task_id: str
    status: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int
