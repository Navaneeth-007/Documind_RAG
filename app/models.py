"""Request/response schemas for the FastAPI app."""
from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: int
    document_title: str
    section_label: Optional[str] = None
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    is_grounded: bool
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
