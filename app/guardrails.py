"""
Guardrails, applied before an LLM call is ever made:

1. Confidence check — if the best reranked chunk's score is below
   `min_relevance_score`, the retrieval simply doesn't contain a good match
   for the question. Answering anyway invites hallucination, so we return
   an honest "I don't know" instead of spending an LLM call rationalizing
   a bad answer.
2. Input sanitization — reject empty/whitespace-only queries and queries
   that look like prompt-injection attempts targeting the system prompt.
"""
import re

from app.config import settings
from app.retrieval.vector_search import RetrievedChunk

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+|the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a(n)?\s+", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(above|previous)", re.IGNORECASE),
    re.compile(r"bypass\s+safety\s+guidelines", re.IGNORECASE),
]

FALLBACK_ANSWER = "I don't have enough information in the provided documents to answer that."


def is_suspicious_query(query: str) -> bool:
    """Check if the user input contains adversarial prompt injection patterns."""
    if not query or not query.strip():
        return False
    return any(pattern.search(query) for pattern in _INJECTION_PATTERNS)


def passes_confidence_threshold(chunks: list[RetrievedChunk], threshold: float | None = None) -> bool:
    """Check if the top retrieved chunk passes the minimum relevance confidence threshold."""
    if not settings.enable_guardrails:
        return bool(chunks)
    if not chunks:
        return False
    min_thresh = threshold if threshold is not None else settings.min_relevance_score
    return chunks[0].score >= min_thresh
