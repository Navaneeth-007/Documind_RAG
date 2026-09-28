"""
Embedding client. Wrapped behind a single function so the embedding provider
can be swapped (OpenAI -> local sentence-transformers -> mock) without touching
ingestion or retrieval code — both only ever call `embed_texts`.
"""
import hashlib
import logging
from typing import Any

from app.config import settings

logger = logging.getLogger("documind.embed")

_openai_client: Any | None = None
_st_model: Any | None = None
_fastembed_model: Any | None = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        kwargs = {"api_key": settings.openai_api_key or "sk-dummy"}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        _openai_client = OpenAI(**kwargs)
    return _openai_client


def _get_fastembed_model():
    global _fastembed_model
    if _fastembed_model is None:
        try:
            from fastembed import TextEmbedding

            model_name = settings.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
            logger.info("Loading FastEmbed (ONNX engine) for '%s' (35MB RAM footprint)...", model_name)
            _fastembed_model = TextEmbedding(model_name=model_name)
        except Exception as e:
            logger.info("FastEmbed unavailable, falling back to SentenceTransformers: %s", e)
            _fastembed_model = False
    return _fastembed_model if _fastembed_model is not False else None


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        model_name = settings.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        _st_model = SentenceTransformer(model_name)
    return _st_model


def _mock_embed(texts: list[str], dim: int) -> list[list[float]]:
    """Deterministic pseudo-embedding for testing without API keys or models."""
    results = []
    for text in texts:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Create normalized floats from hash
        vec = [(b / 255.0) * 2.0 - 1.0 for b in h]
        while len(vec) < dim:
            vec.extend(vec[: min(dim - len(vec), len(vec))])
        vec = vec[:dim]
        # L2 normalize
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        results.append([x / norm for x in vec])
    return results


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Handles OpenAI, FastEmbed ONNX, SentenceTransformers, and Mock providers."""
    if not texts:
        return []

    provider = settings.embedding_provider.lower()

    if provider == "openai":
        client = _get_openai_client()
        response = client.embeddings.create(model=settings.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    elif provider in {"sentence-transformers", "local", "hf", "fastembed"}:
        fe_model = _get_fastembed_model()
        if fe_model is not None:
            embeddings = list(fe_model.embed(texts))
            return [emb.tolist() for emb in embeddings]

        model = _get_st_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    elif provider == "mock":
        return _mock_embed(texts, settings.embedding_dim)

    else:
        raise NotImplementedError(
            f"Embedding provider '{settings.embedding_provider}' is not supported. "
            "Choose 'openai', 'sentence-transformers', or 'mock'."
        )
