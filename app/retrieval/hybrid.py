"""
Hybrid retrieval: fuse dense (vector) and keyword (BM25-style ts_rank) result
lists using Reciprocal Rank Fusion (RRF), rather than trying to normalize and
average two differently-scaled relevance scores directly.

RRF score for a document d given a set of ranked lists R:
    RRF(d) = sum over each list r in R of  1 / (k + rank_r(d))

k (typically 60) dampens the influence of any single list's top result,
so a chunk that ranks decently in *both* dense and keyword search usually
outranks a chunk that's #1 in only one.
"""
from app.ingestion.embed import embed_texts
from app.retrieval.keyword_search import keyword_search
from app.retrieval.vector_search import RetrievedChunk, dense_search

RRF_K = 60


def hybrid_search(query: str, top_k_dense: int, top_k_keyword: int, top_k_final: int) -> list[RetrievedChunk]:
    query_embedding = embed_texts([query])[0]

    dense_results = dense_search(query_embedding, top_k_dense)
    keyword_results = keyword_search(query, top_k_keyword)

    rrf_scores: dict[int, float] = {}
    chunk_lookup: dict[int, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        chunk_lookup[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(keyword_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        chunk_lookup.setdefault(chunk.chunk_id, chunk)

    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k_final]

    fused: list[RetrievedChunk] = []
    for cid in ranked_ids:
        base = chunk_lookup[cid]
        fused.append(
            RetrievedChunk(
                chunk_id=base.chunk_id,
                document_title=base.document_title,
                section_label=base.section_label,
                content=base.content,
                score=rrf_scores[cid],
            )
        )
    return fused
