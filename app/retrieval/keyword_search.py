"""
Keyword retrieval via Postgres full-text search (ts_rank on the precomputed
tsvector column). This is what catches exact identifiers, product codes, and
rare terms that dense embeddings tend to smooth over.
"""
from sqlalchemy import text

from app.db import get_connection
from app.retrieval.vector_search import RetrievedChunk


def keyword_search(query: str, top_k: int) -> list[RetrievedChunk]:
    sql = text(
        """
        SELECT c.id, d.title, c.section_label, c.content,
               ts_rank(c.content_tsv, plainto_tsquery('english', :query)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.content_tsv @@ plainto_tsquery('english', :query)
        ORDER BY score DESC
        LIMIT :top_k
        """
    )
    with get_connection() as conn:
        rows = conn.execute(sql, {"query": query, "top_k": top_k}).fetchall()

    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_title=row.title,
            section_label=row.section_label,
            content=row.content,
            score=float(row.score),
        )
        for row in rows
    ]
