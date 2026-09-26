"""
Embedding client. Wrapped behind a single function so the embedding provider
can be swapped (OpenAI -> local bge-small, etc.) without touching ingestion
or retrieval code — both only ever call `embed_texts`.
"""
from openai import OpenAI

from app.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Batches of >2048 should be chunked by the caller."""
    if settings.embedding_provider == "openai":
        client = _get_client()
        response = client.embeddings.create(model=settings.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    raise NotImplementedError(
        f"Embedding provider '{settings.embedding_provider}' not wired up yet. "
        "Add a branch here (e.g. sentence-transformers for a local model)."
    )
