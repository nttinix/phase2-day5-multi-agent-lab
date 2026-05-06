"""Analyst agent – turns research notes into structured insights."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a critical analyst. Given a research query and research notes, produce a
structured analysis that:
- Lists the 3-5 most important claims or findings.
- Identifies the strength of evidence for each claim (strong / moderate / weak).
- Flags any contradictions, biases, or missing evidence.
- Suggests what additional information would strengthen the analysis.
Format your output with clear headings: Key Claims, Evidence Assessment, Gaps & Caveats.
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights using an LLM."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            raise AgentExecutionError(
                "AnalystAgent requires research_notes in state. Run ResearcherAgent first."
            )

        query = state.request.query
        user_prompt = (
            f"Query: {query}\n\nResearch Notes:\n{state.research_notes}"
        )

        with trace_span("analyst.llm", {"query": query}):
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "analyst.done",
            {
                "analysis_length": len(response.content),
                "cost_usd": response.cost_usd,
            },
        )
        logger.info("AnalystAgent finished. Analysis length=%d", len(response.content))
        return state
