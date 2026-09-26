"""
LLM client, provider-agnostic at the call site. `generate_answer` and `generate_answer_stream`
are the primary interfaces, supporting OpenAI, Anthropic, Groq / OpenAI-compatible,
Bedrock, and Mock providers.
"""
import logging
from dataclasses import dataclass
from typing import Generator, Optional

from app.config import settings
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.models import ChatMessage

logger = logging.getLogger("documind.llm")

# Pricing in USD per 1,000 tokens (for observability & metrics tracking)
PRICING = {
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
}


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING.get(model, {"input": 0.00015, "output": 0.0006})
    return (prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"]


def generate_answer(
    query: str,
    context_chunks: list[str],
    chat_history: Optional[list[ChatMessage]] = None,
) -> LLMResult:
    user_prompt = build_user_prompt(query, context_chunks, chat_history)
    provider = settings.llm_provider.lower()

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        model = settings.llm_model or "claude-3-5-sonnet-20241022"
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

    elif provider in {"openai", "groq"}:
        from openai import OpenAI

        kwargs = {}
        if provider == "groq":
            kwargs["api_key"] = settings.groq_api_key or settings.openai_api_key
            kwargs["base_url"] = "https://api.groq.com/openai/v1"
            model = settings.llm_model or "llama-3.3-70b-versatile"
        else:
            kwargs["api_key"] = settings.openai_api_key
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            model = settings.llm_model or "gpt-4o-mini"

        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else len(user_prompt) // 4
        completion_tokens = response.usage.completion_tokens if response.usage else len(text) // 4

    elif provider == "bedrock":
        import json

        import boto3

        bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
        model = settings.llm_model or "anthropic.claude-3-sonnet-20240229-v1:0"
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            }
        )
        response = bedrock.invoke_model(body=body, modelId=model)
        response_body = json.loads(response.get("body").read())
        text = "".join(b["text"] for b in response_body.get("content", []) if b.get("type") == "text")
        usage = response_body.get("usage", {})
        prompt_tokens = usage.get("input_tokens", len(user_prompt) // 4)
        completion_tokens = usage.get("output_tokens", len(text) // 4)

    elif provider == "mock":
        model = "mock-grounded-v1"
        if not context_chunks:
            text = "I don't have enough information in the provided documents to answer that."
        else:
            first_chunk = context_chunks[0]
            text = f"Based on the provided documentation [Chunk 1]: {first_chunk[:300]}..."
        prompt_tokens = len(user_prompt) // 4
        completion_tokens = len(text) // 4

    else:
        raise NotImplementedError(
            f"LLM provider '{settings.llm_provider}' is not configured. "
            "Supported providers: 'openai', 'anthropic', 'groq', 'bedrock', 'mock'."
        )

    return LLMResult(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=_estimate_cost(model, prompt_tokens, completion_tokens),
    )


def generate_answer_stream(
    query: str,
    context_chunks: list[str],
    chat_history: Optional[list[ChatMessage]] = None,
) -> Generator[str, None, None]:
    """Stream response tokens chunk by chunk."""
    user_prompt = build_user_prompt(query, context_chunks, chat_history)
    provider = settings.llm_provider.lower()

    if provider in {"openai", "groq"}:
        from openai import OpenAI

        kwargs = {}
        if provider == "groq":
            kwargs["api_key"] = settings.groq_api_key or settings.openai_api_key
            kwargs["base_url"] = "https://api.groq.com/openai/v1"
            model = settings.llm_model or "llama-3.3-70b-versatile"
        else:
            kwargs["api_key"] = settings.openai_api_key
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            model = settings.llm_model or "gpt-4o-mini"

        client = OpenAI(**kwargs)
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        model = settings.llm_model or "claude-3-5-sonnet-20241022"
        with client.messages.stream(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text_chunk in stream.text_stream:
                yield text_chunk

    else:
        # Fallback for mock/bedrock: yield full result
        result = generate_answer(query, context_chunks, chat_history)
        yield result.text
