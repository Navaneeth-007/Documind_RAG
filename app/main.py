"""
FastAPI entrypoint. Endpoints are intentionally thin — all logic lives in
app/{retrieval,generation,guardrails,ingestion}, so this file is just wiring
and logging.
"""
import logging
import time

from fastapi import FastAPI, HTTPException
from sqlalchemy import text

from app.config import settings
from app.db import check_connection, get_connection
from app.generation.llm import generate_answer
from app.guardrails import FALLBACK_ANSWER, is_suspicious_query, passes_confidence_threshold
from app.models import Citation, HealthResponse, QueryRequest, QueryResponse
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("documind")

app = FastAPI(title="DocuMind RAG API", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", db_connected=check_connection())


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    if is_suspicious_query(request.query):
        logger.warning("Rejected suspicious query: %r", request.query)
        raise HTTPException(status_code=400, detail="Query rejected by input guardrails.")

    top_k_final = request.top_k or settings.top_k_final

    candidates = hybrid_search(
        query=request.query,
        top_k_dense=settings.top_k_dense,
        top_k_keyword=settings.top_k_keyword,
        top_k_final=max(settings.top_k_dense, settings.top_k_keyword),
    )
    reranked = rerank(request.query, candidates, top_k=top_k_final)

    if not passes_confidence_threshold(reranked):
        latency_ms = int((time.perf_counter() - start) * 1000)
        _log_query(request.query, FALLBACK_ANSWER, [], latency_ms, 0, 0, 0.0)
        return QueryResponse(
            answer=FALLBACK_ANSWER,
            citations=[],
            is_grounded=False,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            estimated_cost_usd=0.0,
        )

    result = generate_answer(request.query, [c.content for c in reranked])
    latency_ms = int((time.perf_counter() - start) * 1000)

    citations = [
        Citation(
            chunk_id=c.chunk_id,
            document_title=c.document_title,
            section_label=c.section_label,
            snippet=c.content[:280],
            score=c.score,
        )
        for c in reranked
    ]

    _log_query(
        request.query,
        result.text,
        [c.chunk_id for c in reranked],
        latency_ms,
        result.prompt_tokens,
        result.completion_tokens,
        result.estimated_cost_usd,
    )

    return QueryResponse(
        answer=result.text,
        citations=citations,
        is_grounded=True,
        latency_ms=latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )


def _log_query(
    query_text: str,
    answer: str,
    chunk_ids: list[int],
    latency_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
) -> None:
    """Best-effort observability logging — a failure here should never break a request."""
    try:
        with get_connection() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO query_logs
                        (query, answer, retrieved_chunk_ids, latency_ms,
                         prompt_tokens, completion_tokens, estimated_cost_usd)
                    VALUES (:query, :answer, :chunk_ids, :latency_ms,
                            :prompt_tokens, :completion_tokens, :cost)
                    """
                ),
                {
                    "query": query_text,
                    "answer": answer,
                    "chunk_ids": chunk_ids,
                    "latency_ms": latency_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": cost,
                },
            )
    except Exception:
        logger.exception("Failed to write query log (non-fatal)")
