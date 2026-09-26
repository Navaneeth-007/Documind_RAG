"""Prompt templates. Kept as plain strings + a builder function so they're
easy to version and A/B test without digging through the LLM client code."""

SYSTEM_PROMPT = """You are DocuMind, a document Q&A assistant. Answer the user's \
question using ONLY the provided context chunks. Follow these rules strictly:

1. If the answer is not contained in the context, say "I don't have enough \
information in the provided documents to answer that." Do not guess.
2. Every factual claim must be traceable to a specific context chunk.
3. Be concise. Do not repeat the question or add unrequested caveats.
4. Do not use outside knowledge, even if you believe you know the answer.
"""


def build_user_prompt(query: str, context_chunks: list[str]) -> str:
    context_block = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above. Reference chunk numbers "
        "(e.g. \"[Chunk 2]\") where relevant."
    )
