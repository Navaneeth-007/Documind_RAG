"""
Centralized app configuration, loaded from environment variables (.env).
Using pydantic-settings so config is validated at startup rather than
failing deep inside a request handler with a confusing KeyError.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql://documind:documind@localhost:5432/documind"

    # LLM provider
    llm_provider: str = "anthropic"  # anthropic | openai | bedrock
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    # Embeddings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # Retrieval tuning
    top_k_dense: int = 20
    top_k_keyword: int = 20
    top_k_final: int = 5
    reranker_model: str = "BAAI/bge-reranker-base"

    # Guardrails
    min_relevance_score: float = 0.35  # below this, respond "I don't know"

    # App
    app_env: str = "development"
    log_level: str = "INFO"


settings = Settings()
