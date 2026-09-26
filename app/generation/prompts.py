"""
Prompt templates. Kept as plain strings + a builder function so they're
easy to version and A/B test without digging through the LLM client code.
"""
from typing import Optional

from app.models import ChatMessage

SYSTEM_PROMPT = """You are DocuMind, a precision document Q&A assistant. Answer the user's \
question using ONLY the provided context chunks. Follow these rules strictly:

1. If the answer is not contained in the context, say "I don't have enough \
information in the provided documents to answer that." Do not guess or extrapolate.
2. Every factual claim must be traceable to a specific context chunk.
3. Be concise and accurate. Do not repeat the question or add unrequested caveats.
4. Do not use outside knowledge, even if you believe you know the answer.
5. Whenever quoting or referencing facts, reference the chunk (e.g. "[Chunk 1]").
"""


def build_user_prompt(
    query: str,
    context_chunks: list[str],
    chat_history: Optional[list[ChatMessage]] = None,
) -> str:
    context_block = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    history_block = ""
    if chat_history:
        formatted_history = []
        for msg in chat_history[-6:]:  # keep last 6 turns for context
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted_history.append(f"{role_label}: {msg.content}")
        history_block = "Recent Conversation History:\n" + "\n".join(formatted_history) + "\n\n"

    return (
        f"{history_block}"
        f"Context Chunks:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context chunks above. Reference chunk numbers (e.g. \"[Chunk 1]\") where appropriate."
    )
