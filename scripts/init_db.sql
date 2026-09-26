-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table: one row per source file
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    source_path TEXT NOT NULL,
    title TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- Chunks table: one row per chunk, with embedding + full-text search vector
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    section_label TEXT,
    embedding VECTOR(1536),
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMP DEFAULT now()
);

-- Index for dense vector similarity search (cosine distance)
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for keyword/full-text search
CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
    ON chunks USING GIN (content_tsv);

-- Query log for observability (latency, cost, tokens per query)
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
