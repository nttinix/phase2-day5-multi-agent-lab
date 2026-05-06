"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# Cost per 1 000 tokens (USD) for gpt-4o-mini – update if model changes
_COST_PER_1K_INPUT = 0.000150
_COST_PER_1K_OUTPUT = 0.000600


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client backed by OpenAI."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = self._build_client()

    def _build_client(self) -> object:
        try:
            from openai import OpenAI  # type: ignore[import-untyped]

            api_key = self._settings.openai_api_key
            if not api_key:
                logger.warning("OPENAI_API_KEY not set – LLMClient will fail on real calls.")
                return None

            client = OpenAI(api_key=api_key)

            # Wrap với LangSmith nếu có API key — tự động gửi trace lên LangSmith
            if self._settings.langsmith_api_key:
                try:
                    from langsmith.wrappers import wrap_openai  # type: ignore[import-untyped]
                    client = wrap_openai(client)
                    logger.info("LangSmith tracing enabled for LLMClient.")
                except ImportError:
                    logger.debug("langsmith not installed – tracing disabled.")

            return client
        except ImportError:
            logger.warning("openai package not installed – LLMClient unavailable.")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion via OpenAI chat completions API."""

        if self._client is None:
            raise RuntimeError(
                "LLMClient is not configured. Install 'openai' and set OPENAI_API_KEY."
            )

        from openai import OpenAI  # type: ignore[import-untyped]

        client: OpenAI = self._client  # type: ignore[assignment]
        model = self._settings.openai_model

        logger.debug("LLMClient.complete model=%s", model)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self._settings.timeout_seconds,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost: float | None = None
        if input_tokens is not None and output_tokens is not None:
            cost = (input_tokens / 1000) * _COST_PER_1K_INPUT + (
                output_tokens / 1000
            ) * _COST_PER_1K_OUTPUT

        logger.debug(
            "LLMClient response tokens in=%s out=%s cost=%.6f",
            input_tokens,
            output_tokens,
            cost or 0,
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
