"""
Guardrails, applied before an LLM call is ever made:

1. Confidence check — if the best reranked chunk's score is below
   `min_relevance_score`, the retrieval simply doesn't contain a good match
   for the question. Answering anyway invites hallucination, so we return
   an honest "I don't know" instead of spending an LLM call rationalizing
   a bad answer.
2. Basic input sanitation — reject empty/whitespace-only queries and queries
   that look like prompt-injection attempts targeting the system prompt.
"""
import re

from app.config import settings
from app.retrieval.vector_search import RetrievedChunk

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?previous instructions", re.IGNORECASE),
    re.compile(r"reveal (your |the )?system prompt", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
]

FALLBACK_ANSWER = "I don't have enough information in the provided documents to answer that."


def is_suspicious_query(query: str) -> bool:
    return any(pattern.search(query) for pattern in _INJECTION_PATTERNS)


def passes_confidence_threshold(chunks: list[RetrievedChunk]) -> bool:
    if not chunks:
        return False
    return chunks[0].score >= settings.min_relevance_score
