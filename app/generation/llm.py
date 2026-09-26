"""
LLM client, provider-agnostic at the call site. `generate_answer` is the only
function the rest of the app depends on, so swapping Anthropic <-> OpenAI <->
Bedrock only means editing this file.
"""
from dataclasses import dataclass

from app.config import settings
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt

# Rough per-model pricing for cost estimation/logging (USD per 1K tokens).
# Update these if you change models — they're for observability, not billing.
PRICING = {
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"]


def generate_answer(query: str, context_chunks: list[str]) -> LLMResult:
    user_prompt = build_user_prompt(query, context_chunks)

    if settings.llm_provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        model = "claude-sonnet-4-6"
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

    elif settings.llm_provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        model = "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

    else:
        raise NotImplementedError(
            f"LLM provider '{settings.llm_provider}' not wired up yet. "
            "Add a branch here for Bedrock or another provider."
        )

    return LLMResult(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=_estimate_cost(model, prompt_tokens, completion_tokens),
    )
