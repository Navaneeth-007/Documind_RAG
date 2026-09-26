# DocuMind — Production-Grade RAG Assistant

A retrieval-augmented generation system for querying a document corpus, built with hybrid
retrieval (dense + keyword), cross-encoder reranking, source citations, guardrails, and a
quantitative evaluation harness — not just a "chat with your PDF" demo.

**Live demo:** _add your deployed link here once hosted_
**Architecture diagram:** _add a screenshot/diagram here (see `docs/architecture.png` placeholder)_

---

## Why this exists

Most portfolio RAG projects are a single notebook: chunk → embed → cosine similarity → prompt
the LLM. That works on a slide, but falls apart on real queries — synonyms the embedding model
doesn't capture, questions needing exact terms (IDs, section numbers), and no way to know if an
answer is actually grounded in the source.

DocuMind is built the way I'd build this at work:

- **Hybrid retrieval** — dense vector search (pgvector) fused with keyword/BM25 search via
  reciprocal rank fusion, so exact-term queries don't get lost in embedding space.
- **Reranking** — a cross-encoder reranks the fused candidate set before it reaches the LLM.
- **Citations** — every answer is grounded to specific source chunks; the API returns them
  alongside the response.
- **Guardrails** — out-of-scope questions and low-confidence retrievals get an honest
  "I don't know" instead of a hallucinated answer.
- **Evaluation harness** — a golden Q&A set scored for faithfulness, answer relevance, and
  retrieval precision@k, with results tracked in this README (see below).
- **Observability** — every query is logged with latency, token usage, and estimated cost.
- **Tested, containerized, CI'd** — pytest for the retrieval/generation logic, GitHub Actions
  running lint + tests on push, and a one-command Docker deploy.

## Architecture

```
                     ┌──────────────────┐
   User ──────────▶  │   Streamlit UI    │
                     └────────┬─────────┘
                              │ HTTP
                     ┌────────▼─────────┐
                     │   FastAPI app     │
                     │  (app/main.py)    │
                     └────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌────────────┐  ┌─────────────┐  ┌────────────┐
       │ Guardrails │  │  Retrieval  │  │ Generation │
       │  (scope +  │  │  (hybrid +  │  │  (prompt + │
       │ confidence)│  │  rerank)    │  │  LLM call) │
       └────────────┘  └──────┬──────┘  └──────┬─────┘
                              │                │
                       ┌──────▼──────┐         │
                       │  pgvector   │         │
                       │ (Postgres)  │         │
                       └─────────────┘         │
                                                ▼
                                       Response + citations
                                       + logged latency/cost
```

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain / LangGraph |
| Vector store | Postgres + pgvector |
| Keyword search | Postgres full-text search (`tsvector`) |
| Embeddings | `text-embedding-3-small` (swappable for local `bge-small`) |
| Reranker | `bge-reranker-base` (cross-encoder, local) or Cohere rerank |
| LLM | Anthropic API / Amazon Bedrock (swappable via `app/generation/llm.py`) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Eval | Custom faithfulness/relevance scorer (`eval/`) modeled on RAGAS metrics |
| Deployment | Docker Compose locally; Render/Fly.io/ECS for hosting |

## Design decisions (worth reading before you judge the code)

- **Chunking:** recursive character splitting with a 500-token target and 15% overlap, not fixed
  boundaries — see `app/ingestion/chunker.py` for the reasoning in comments. Section headers are
  preserved in chunk metadata so citations can reference "Section 3.2" rather than a raw offset.
- **Hybrid over pure dense retrieval:** dense embeddings miss exact identifiers, product codes,
  and rare terms. Reciprocal rank fusion (`app/retrieval/hybrid.py`) combines both without needing
  a learned fusion model.
- **Reranking is a separate stage, not baked into retrieval:** keeps retrieval fast (broad recall)
  and reranking precise (top-k precision), and either can be swapped independently.
- **Guardrails before generation, not after:** rejecting an out-of-scope query before spending an
  LLM call saves cost and avoids the model rationalizing an answer it shouldn't give.

## Evaluation results

Run `python eval/run_eval.py` after ingesting your corpus. Results are written to
`eval/results.json` and should be pasted here, e.g.:

| Metric | Score |
|---|---|
| Faithfulness | _fill in_ |
| Answer relevance | _fill in_ |
| Retrieval precision@5 | _fill in_ |
| Avg. latency (s) | _fill in_ |
| Avg. cost per query ($) | _fill in_ |

## Getting started

### 1. Prerequisites
- Docker + Docker Compose
- Python 3.11+
- An LLM API key (Anthropic, OpenAI, or AWS Bedrock credentials)

### 2. Setup
```bash
cp .env.example .env        # fill in your API keys and DB URL
docker compose up -d db     # starts Postgres + pgvector
pip install -r requirements.txt
```

### 3. Ingest your documents
```bash
python scripts/ingest_documents.py --source ./data/sample_docs
```

### 4. Run the API
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Run the frontend
```bash
streamlit run frontend/streamlit_app.py
```

### 6. Run tests
```bash
pytest tests/ -v
```

### 7. Run the eval harness
```bash
python eval/run_eval.py
```

### 8. Full stack via Docker
```bash
docker compose up --build
```

## Project structure
```
documind-rag/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py                # settings via env vars
│   ├── models.py                 # Pydantic request/response schemas
│   ├── db.py                     # Postgres/pgvector connection
│   ├── guardrails.py             # scope + confidence checks
│   ├── ingestion/                # loading, chunking, embedding
│   ├── retrieval/                # vector search, keyword search, fusion, reranking
│   └── generation/                # prompt templates + LLM client
├── frontend/streamlit_app.py     # chat UI
├── eval/                          # golden dataset + evaluation scripts
├── scripts/ingest_documents.py    # CLI ingestion entrypoint
├── tests/                        # pytest suite
├── .github/workflows/ci.yml      # lint + test on push
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Roadmap
- [ ] Swap in a real deployed frontend (Next.js) once Streamlit version is validated
- [ ] Add streaming responses (SSE) from the FastAPI backend
- [ ] Add conversation memory / multi-turn context
- [ ] Add a small LLMOps dashboard (token cost + latency trends over time)

## License
MIT
