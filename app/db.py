"""
Thin database access layer. Deliberately raw-SQL-via-SQLAlchemy-core rather than
a full ORM: for a retrieval-heavy app the queries (vector similarity, full-text
search, RRF fusion) are easier to reason about and tune as SQL than through an
ORM abstraction layer.
"""
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import settings

logger = logging.getLogger("documind.db")
_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
    return _engine


@contextmanager
def get_connection():
    engine = get_engine()
    conn = engine.connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_connection() -> bool:
    try:
        with get_connection() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"Database connection check failed: {e}")
        return False


def init_database() -> None:
    """Auto-initialize database tables and pgvector extension on startup."""
    try:
        with get_connection() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        source_path TEXT NOT NULL,
                        title TEXT,
                        created_at TIMESTAMP DEFAULT now()
                    );
                    """
                )
            )
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id SERIAL PRIMARY KEY,
                        document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        section_label TEXT,
                        embedding VECTOR({settings.embedding_dim}),
                        content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                        created_at TIMESTAMP DEFAULT now()
                    );
                    """
                )
            )
            # Create indices
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                        ON chunks USING hnsw (embedding vector_cosine_ops);
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
                        ON chunks USING GIN (content_tsv);
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS query_logs (
                        id SERIAL PRIMARY KEY,
                        query TEXT NOT NULL,
                        answer TEXT,
                        retrieved_chunk_ids INTEGER[],
                        latency_ms INTEGER,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        estimated_cost_usd NUMERIC(10, 6),
                        created_at TIMESTAMP DEFAULT now()
                    );
                    """
                )
            )
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.warning(f"Database schema auto-init failed or skipped (may be unready): {e}")
