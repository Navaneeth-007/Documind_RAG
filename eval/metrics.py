"""
Lightweight evaluation metrics, modeled on RAGAS's definitions but implemented
without the extra dependency so the eval harness stays simple to read and
modify:

- Retrieval precision@k: of the chunks retrieved, what fraction come from a
  document the golden set marked as actually relevant to the question.
- Answer relevance: keyword-overlap proxy — does the generated answer contain
  the expected key phrase(s)? (A cheap stand-in for embedding-similarity-based
  relevance; swap in an LLM-as-judge call for a stronger signal if needed.)
- Faithfulness: does every sentence in the answer have reasonable lexical
  overlap with at least one retrieved chunk? Flags likely hallucinations
  where the model asserted something not present in context.
"""
import re


def retrieval_precision_at_k(retrieved_titles: list[str], relevant_titles: list[str]) -> float:
    if not retrieved_titles:
        return 0.0
    if not relevant_titles:
        # No relevant docs expected (e.g. an intentionally unanswerable question) —
        # precision is trivially perfect if nothing irrelevant leaked through as "confident".
        return 1.0
    hits = sum(1 for t in retrieved_titles if t in relevant_titles)
    return hits / len(retrieved_titles)


def answer_relevance(answer: str, expected_phrases: list[str]) -> float:
    if not expected_phrases:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for phrase in expected_phrases if phrase.lower() in answer_lower)
    return hits / len(expected_phrases)


def _sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def faithfulness(answer: str, context_chunks: list[str]) -> float:
    sentences = _sentence_split(answer)
    if not sentences:
        return 1.0

    context_words = set()
    for chunk in context_chunks:
        context_words.update(re.findall(r"\w+", chunk.lower()))

    grounded_count = 0
    for sentence in sentences:
        sentence_words = set(re.findall(r"\w+", sentence.lower()))
        if not sentence_words:
            continue
        overlap = len(sentence_words & context_words) / max(len(sentence_words), 1)
        if overlap >= 0.3:
            grounded_count += 1

    return grounded_count / len(sentences)
