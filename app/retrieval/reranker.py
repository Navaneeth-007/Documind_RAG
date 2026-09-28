"""
Reranking stage, kept separate from retrieval on purpose: retrieval (hybrid
search) optimizes for recall over a broad candidate set cheaply, while
reranking optimizes for precision on a small candidate set using a heavier
cross-encoder model that scores the (query, chunk) pair jointly rather than
comparing independently-computed embeddings.
"""
import logging
import math
from functools import lru_cache

from app.config import settings
from app.retrieval.vector_search import RetrievedChunk

logger = logging.getLogger("documind.reranker")


@lru_cache(maxsize=1)
def _get_reranker():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(settings.reranker_model)
    except Exception as e:
        logger.warning(f"Could not load CrossEncoder '{settings.reranker_model}': {e}. Falling back to RRF scores.")
        return None


def _sigmoid(x: float) -> float:
    """Map raw logits to [0.0, 1.0] confidence score."""
    if x < -40.0:
        return 0.0
    elif x > 40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not candidates:
        return []

    if not settings.use_reranker:
        return candidates[:top_k]

    model = _get_reranker()
    if model is None:
        # Fallback to candidates as ranked by hybrid search
        return candidates[:top_k]

    try:
        pairs = [(query, c.content) for c in candidates]
        raw_scores = model.predict(pairs)

        scored = [(chunk, _sigmoid(float(score))) for chunk, score in zip(candidates, raw_scores)]
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
    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Returning hybrid candidates.")
        return candidates[:top_k]
