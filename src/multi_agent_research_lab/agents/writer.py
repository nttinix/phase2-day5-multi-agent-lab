"""Writer agent – produces the final answer from research and analysis notes."""

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
You are a skilled technical writer. Given a research query, research notes, and an
analysis, write a clear, well-structured response that:
- Directly answers the query.
- Integrates key findings and evidence from the research and analysis.
- Cites sources using [1], [2], … notation where relevant.
- Is appropriate for the stated audience.
- Is between 400-600 words unless the query specifies otherwise.
Do NOT add information not present in the provided notes.
"""


class WriterAgent(BaseAgent):
    """Synthesises a final answer with citations from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            raise AgentExecutionError(
                "WriterAgent requires research_notes in state. Run ResearcherAgent first."
            )

        query = state.request.query
        audience = state.request.audience
        analysis_section = (
            f"\n\nAnalysis Notes:\n{state.analysis_notes}"
            if state.analysis_notes
            else ""
        )

        # Build source reference list for the prompt
        source_refs = "\n".join(
            f"[{i + 1}] {s.title} – {s.url or 'no url'}"
            for i, s in enumerate(state.sources)
        )

        user_prompt = (
            f"Query: {query}\n"
            f"Audience: {audience}\n\n"
            f"Research Notes:\n{state.research_notes}"
            f"{analysis_section}\n\n"
            f"Source References:\n{source_refs}"
        )

        with trace_span("writer.llm", {"query": query}):
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "writer.done",
            {
                "answer_length": len(response.content),
                "cost_usd": response.cost_usd,
            },
        )
        logger.info("WriterAgent finished. Answer length=%d", len(response.content))
        return state
