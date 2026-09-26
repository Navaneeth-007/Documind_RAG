"""Request/response schemas for the FastAPI app."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant' or 'system'")
    content: str = Field(..., description="Message text")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of context chunks")
    chat_history: Optional[list[ChatMessage]] = Field(default=None, description="Prior conversation context")
    stream: bool = Field(default=False, description="Whether to stream response tokens")


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


class DocumentInfo(BaseModel):
    id: int
    title: str
    source_path: str
    chunk_count: int
    created_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int


class IngestTextRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    source_path: Optional[str] = "direct_upload"


class QueryLogEntry(BaseModel):
    id: int
    query: str
    answer: Optional[str]
    retrieved_chunk_ids: Optional[list[int]] = None
    latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    created_at: Optional[datetime] = None


class AnalyticsSummary(BaseModel):
    total_queries: int
    avg_latency_ms: float
    total_cost_usd: float
    grounded_queries: int
    ungrounded_queries: int
    grounding_rate_pct: float
    total_prompt_tokens: int
    total_completion_tokens: int
    recent_logs: list[QueryLogEntry]


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    version: str
    llm_provider: str
    embedding_provider: str
    reranker_model: str
