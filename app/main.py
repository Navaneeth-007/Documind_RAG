"""
FastAPI entrypoint. Exposes clean REST endpoints for querying, streaming,
document management, and LLMOps observability metrics.
"""
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from app.config import settings
from app.db import check_connection, get_connection, init_database
from app.generation.llm import generate_answer, generate_answer_stream
from app.guardrails import FALLBACK_ANSWER, is_suspicious_query, passes_confidence_threshold
from app.ingestion.chunker import chunk_document
from app.ingestion.embed import embed_texts
from app.ingestion.loader import RawDocument, load_document_from_bytes
from app.models import (
    AnalyticsSummary,
    Citation,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryLogEntry,
    QueryRequest,
    QueryResponse,
)
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("documind")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing DocuMind RAG database schema...")
    init_database()
    try:
        logger.info("Warming up embedding and reranker models...")
        embed_texts(["warmup query"])
        from app.retrieval.hybrid import ChunkCandidate
        rerank("warmup", [ChunkCandidate(chunk_id=1, document_id=1, document_title="t", section_label="s", content="c", score=1.0)], top_k=1)
        logger.info("Model warmup complete!")
    except Exception as e:
        logger.warning("Warmup warning (non-fatal): %s", e)
    yield
    logger.info("Shutting down DocuMind RAG API...")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
)

# Enable CORS for local development and web hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    db_ok = check_connection()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db_connected=db_ok,
        version=settings.api_version,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        reranker_model=settings.reranker_model if settings.use_reranker else "none",
    )


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

    result = generate_answer(
        query=request.query,
        context_chunks=[c.content for c in reranked],
        chat_history=request.chat_history,
    )
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


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """Stream answer tokens over Server-Sent Events (SSE)."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    if is_suspicious_query(request.query):
        raise HTTPException(status_code=400, detail="Query rejected by input guardrails.")

    top_k_final = request.top_k or settings.top_k_final
    candidates = hybrid_search(
        query=request.query,
        top_k_dense=settings.top_k_dense,
        top_k_keyword=settings.top_k_keyword,
        top_k_final=max(settings.top_k_dense, settings.top_k_keyword),
    )
    reranked = rerank(request.query, candidates, top_k=top_k_final)

    def event_generator():
        if not passes_confidence_threshold(reranked):
            yield f"data: {json.dumps({'type': 'content', 'token': FALLBACK_ANSWER})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'is_grounded': False, 'citations': []})}\n\n"
            return

        citations = [
            {
                "chunk_id": c.chunk_id,
                "document_title": c.document_title,
                "section_label": c.section_label,
                "snippet": c.content[:280],
                "score": c.score,
            }
            for c in reranked
        ]
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

        full_answer = []
        for token in generate_answer_stream(
            query=request.query,
            context_chunks=[c.content for c in reranked],
            chat_history=request.chat_history,
        ):
            full_answer.append(token)
            yield f"data: {json.dumps({'type': 'content', 'token': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'is_grounded': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a file (.txt, .md, .pdf, .json, .csv) and ingest it into the RAG corpus."""
    content = await file.read()
    raw_doc = load_document_from_bytes(file.filename, content)
    if not raw_doc:
        raise HTTPException(status_code=400, detail=f"Could not parse document '{file.filename}'.")

    chunks_count = _save_document_and_chunks(raw_doc)
    return IngestResponse(documents_ingested=1, chunks_created=chunks_count)


@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(request: IngestTextRequest) -> IngestResponse:
    """Directly ingest a raw text block with a title."""
    raw_doc = RawDocument(
        source_path=request.source_path or "direct_upload",
        title=request.title,
        text=request.content,
    )
    chunks_count = _save_document_and_chunks(raw_doc)
    return IngestResponse(documents_ingested=1, chunks_created=chunks_count)


@app.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """List all ingested documents and their chunk counts."""
    sql = text(
        """
        SELECT d.id, d.title, d.source_path, d.created_at, COUNT(c.id) AS chunk_count
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.id, d.title, d.source_path, d.created_at
        ORDER BY d.id DESC
        """
    )
    try:
        with get_connection() as conn:
            rows = conn.execute(sql).fetchall()

        docs = [
            DocumentInfo(
                id=row.id,
                title=row.title or "Untitled",
                source_path=row.source_path,
                chunk_count=int(row.chunk_count),
                created_at=row.created_at,
            )
            for row in rows
        ]
        total_chunks = sum(d.chunk_count for d in docs)
        return DocumentListResponse(
            documents=docs,
            total_documents=len(docs),
            total_chunks=total_chunks,
        )
    except Exception:
        logger.exception("Failed to query documents")
        return DocumentListResponse(documents=[], total_documents=0, total_chunks=0)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    """Delete a document and its associated chunks from the vector store."""
    try:
        with get_connection() as conn:
            res = conn.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "success", "message": f"Document {doc_id} deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics", response_model=AnalyticsSummary)
def get_analytics() -> AnalyticsSummary:
    """Retrieve observability metrics, token costs, latency stats, and query logs."""
    try:
        with get_connection() as conn:
            summary_sql = text(
                """
                SELECT
                    COUNT(*) AS total_queries,
                    COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                    COALESCE(SUM(estimated_cost_usd), 0) AS total_cost_usd,
                    COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                    COUNT(CASE WHEN answer != :fallback THEN 1 END) AS grounded_queries,
                    COUNT(CASE WHEN answer = :fallback THEN 1 END) AS ungrounded_queries
                FROM query_logs
                """
            )
            summary_row = conn.execute(summary_sql, {"fallback": FALLBACK_ANSWER}).fetchone()

            logs_sql = text(
                """
                SELECT id, query, answer, retrieved_chunk_ids, latency_ms,
                       prompt_tokens, completion_tokens, estimated_cost_usd, created_at
                FROM query_logs
                ORDER BY id DESC
                LIMIT 50
                """
            )
            log_rows = conn.execute(logs_sql).fetchall()

        total_q = summary_row.total_queries if summary_row else 0
        grounded_q = summary_row.grounded_queries if summary_row else 0
        grounding_rate = (grounded_q / total_q * 100.0) if total_q > 0 else 100.0

        recent_logs = [
            QueryLogEntry(
                id=r.id,
                query=r.query,
                answer=r.answer,
                retrieved_chunk_ids=r.retrieved_chunk_ids,
                latency_ms=r.latency_ms,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
                estimated_cost_usd=float(r.estimated_cost_usd or 0),
                created_at=r.created_at,
            )
            for r in log_rows
        ]

        return AnalyticsSummary(
            total_queries=total_q,
            avg_latency_ms=float(summary_row.avg_latency_ms) if summary_row else 0.0,
            total_cost_usd=float(summary_row.total_cost_usd) if summary_row else 0.0,
            grounded_queries=grounded_q,
            ungrounded_queries=summary_row.ungrounded_queries if summary_row else 0,
            grounding_rate_pct=round(grounding_rate, 2),
            total_prompt_tokens=int(summary_row.total_prompt_tokens) if summary_row else 0,
            total_completion_tokens=int(summary_row.total_completion_tokens) if summary_row else 0,
            recent_logs=recent_logs,
        )
    except Exception:
        logger.exception("Failed to retrieve analytics")
        return AnalyticsSummary(
            total_queries=0,
            avg_latency_ms=0.0,
            total_cost_usd=0.0,
            grounded_queries=0,
            ungrounded_queries=0,
            grounding_rate_pct=100.0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            recent_logs=[],
        )


def _save_document_and_chunks(doc: RawDocument) -> int:
    """Helper to chunk, embed, and store a document in Postgres."""
    with get_connection() as conn:
        doc_id = conn.execute(
            text("INSERT INTO documents (source_path, title) VALUES (:path, :title) RETURNING id"),
            {"path": doc.source_path, "title": doc.title},
        ).scalar_one()

        chunks = chunk_document(doc)
        if not chunks:
            return 0

        embeddings = embed_texts([c.content for c in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            conn.execute(
                text(
                    """
                    INSERT INTO chunks
                        (document_id, chunk_index, content, section_label, embedding)
                    VALUES (:doc_id, :idx, :content, :label, :embedding)
                    """
                ),
                {
                    "doc_id": doc_id,
                    "idx": chunk.chunk_index,
                    "content": chunk.content,
                    "label": chunk.section_label,
                    "embedding": embedding,
                },
            )
    return len(chunks)


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
