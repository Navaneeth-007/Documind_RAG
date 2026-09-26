"""Dense retrieval: cosine similarity search over pgvector embeddings."""
from dataclasses import dataclass

from sqlalchemy import text

from app.db import get_connection


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_title: str
    section_label: str | None
    content: str
    score: float


def dense_search(query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
    """
    Returns the top_k chunks by cosine similarity. pgvector's `<=>` operator
    returns cosine *distance*, so we convert to a similarity score (1 - distance)
    for consistency with the keyword search's relevance scoring.
    """
    sql = text(
        """
        SELECT c.id, d.title, c.section_label, c.content,
               1 - (c.embedding <=> (:embedding)::vector) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> (:embedding)::vector
        LIMIT :top_k
        """
    )
    with get_connection() as conn:
        rows = conn.execute(sql, {"embedding": query_embedding, "top_k": top_k}).fetchall()

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
