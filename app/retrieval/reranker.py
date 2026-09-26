"""
Reranking stage, kept separate from retrieval on purpose: retrieval (hybrid
search) optimizes for recall over a broad candidate set cheaply, while
reranking optimizes for precision on a small candidate set using a heavier
cross-encoder model that scores the (query, chunk) pair jointly rather than
comparing independently-computed embeddings.

Loaded lazily and cached at module level since loading a cross-encoder is
expensive and we don't want to pay that cost on every request.
"""
from functools import lru_cache

from app.config import settings
from app.retrieval.vector_search import RetrievedChunk


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, c.content) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    reranked = []
    for chunk, score in scored[:top_k]:
        reranked.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_title=chunk.document_title,
                section_label=chunk.section_label,
                content=chunk.content,
                score=float(score),
            )
        )
    return reranked
