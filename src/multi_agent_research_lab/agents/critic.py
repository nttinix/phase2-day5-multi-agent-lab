"""Critic agent – optional fact-checking and quality review."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a rigorous fact-checker and quality reviewer. Given a research query, the
research notes, and the final answer, evaluate the answer on:
1. Factual accuracy – are claims supported by the research notes?
2. Citation coverage – what fraction of key claims have a source reference?
3. Completeness – does the answer address all parts of the query?
4. Clarity – is the answer well-structured and easy to follow?

Provide a brief critique (100-200 words) and a quality score from 0 to 10.
Format: SCORE: <number>\nCRITIQUE: <text>
"""


class CriticAgent(BaseAgent):
    """Validates the final answer and appends a quality score to state."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.final_answer:
            logger.warning("CriticAgent skipped – no final_answer in state.")
            return state

        query = state.request.query
        user_prompt = (
            f"Query: {query}\n\n"
            f"Research Notes:\n{state.research_notes or '(none)'}\n\n"
            f"Final Answer:\n{state.final_answer}"
        )

        with trace_span("critic.llm", {"query": query}):
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

        # Parse score from response
        quality_score: float | None = None
        for line in response.content.splitlines():
            if line.upper().startswith("SCORE:"):
                try:
                    quality_score = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
                break

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={
                    "quality_score": quality_score,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "critic.done",
            {"quality_score": quality_score, "cost_usd": response.cost_usd},
        )
        logger.info("CriticAgent finished. Quality score=%s", quality_score)
        return state
