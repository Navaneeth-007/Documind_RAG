from unittest.mock import patch

from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_search import RetrievedChunk


def _chunk(cid, score=1.0):
    return RetrievedChunk(chunk_id=cid, document_title=f"doc-{cid}", section_label=None, content="text", score=score)


@patch("app.retrieval.hybrid.embed_texts", return_value=[[0.1, 0.2, 0.3]])
@patch("app.retrieval.hybrid.keyword_search")
@patch("app.retrieval.hybrid.dense_search")
def test_chunk_ranked_first_in_both_lists_wins(mock_dense, mock_keyword, mock_embed):
    mock_dense.return_value = [_chunk(1), _chunk(2), _chunk(3)]
    mock_keyword.return_value = [_chunk(1), _chunk(4), _chunk(5)]

    results = hybrid_search("query", top_k_dense=3, top_k_keyword=3, top_k_final=5)

    assert results[0].chunk_id == 1  # ranked #1 in both lists -> highest RRF score


@patch("app.retrieval.hybrid.embed_texts", return_value=[[0.1, 0.2, 0.3]])
@patch("app.retrieval.hybrid.keyword_search")
@patch("app.retrieval.hybrid.dense_search")
def test_result_count_respects_top_k_final(mock_dense, mock_keyword, mock_embed):
    mock_dense.return_value = [_chunk(i) for i in range(10)]
    mock_keyword.return_value = [_chunk(i) for i in range(10, 20)]

    results = hybrid_search("query", top_k_dense=10, top_k_keyword=10, top_k_final=5)

    assert len(results) == 5


@patch("app.retrieval.hybrid.embed_texts", return_value=[[0.1, 0.2, 0.3]])
@patch("app.retrieval.hybrid.keyword_search", return_value=[])
@patch("app.retrieval.hybrid.dense_search")
def test_handles_empty_keyword_results(mock_dense, mock_keyword, mock_embed):
    mock_dense.return_value = [_chunk(1), _chunk(2)]

    results = hybrid_search("query", top_k_dense=2, top_k_keyword=2, top_k_final=5)

    assert len(results) == 2
