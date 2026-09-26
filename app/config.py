"""
Centralized app configuration, loaded from environment variables (.env).
Using pydantic-settings so config is validated at startup rather than
failing deep inside a request handler with a confusing KeyError.
"""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql://documind:documind@localhost:5432/documind"

    # LLM provider: "anthropic" | "openai" | "groq" | "bedrock" | "mock"
    llm_provider: str = "openai"
    llm_model: Optional[str] = None
    openai_api_key: str = ""
    openai_base_url: Optional[str] = None
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    # Embeddings: "openai" | "sentence-transformers" | "mock"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Retrieval tuning
    top_k_dense: int = 20
    top_k_keyword: int = 20
    top_k_final: int = 5
    reranker_model: str = "BAAI/bge-reranker-base"
    use_reranker: bool = True

    # Guardrails
    min_relevance_score: float = 0.30  # below this, respond "I don't know"
    enable_guardrails: bool = True

    # App & Observability
    app_env: str = "development"
    log_level: str = "INFO"
    api_title: str = "DocuMind RAG API"
    api_version: str = "1.0.0"


settings = Settings()
