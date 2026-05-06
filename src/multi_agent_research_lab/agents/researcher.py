"""Researcher agent – collects sources and writes concise research notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a meticulous research assistant. Given a query and a set of source snippets,
produce concise, factual research notes that:
- Summarise the key findings from the sources.
- Highlight agreements and contradictions between sources.
- Note any important gaps or caveats.
- Keep each point attributable to a source (use [1], [2], … notation).
Write in plain prose, 200-400 words.
"""


class ResearcherAgent(BaseAgent):
    """Collects sources via search and synthesises research notes with an LLM."""

    name = "researcher"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._search = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        query = state.request.query
        max_sources = state.request.max_sources

        with trace_span("researcher.search", {"query": query}) as span:
            sources = self._search.search(query, max_results=max_sources)
            span["attributes"]["num_sources"] = len(sources)

        state.sources = sources
        state.add_trace_event("researcher.search", {"num_sources": len(sources)})
        logger.info("ResearcherAgent found %d sources.", len(sources))

        # Build a numbered source list for the LLM prompt
        source_block = "\n".join(
            f"[{i + 1}] {s.title}: {s.snippet}" for i, s in enumerate(sources)
        )
        user_prompt = f"Query: {query}\n\nSources:\n{source_block}"

        with trace_span("researcher.llm", {"model": "llm_client"}):
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher.done",
            {
                "notes_length": len(response.content),
                "cost_usd": response.cost_usd,
            },
        )
        logger.info("ResearcherAgent finished. Notes length=%d", len(response.content))
        return state
