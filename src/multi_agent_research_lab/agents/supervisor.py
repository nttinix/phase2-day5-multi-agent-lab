"""Supervisor / router – decides which worker runs next and when to stop."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

# Sentinel value stored in route_history to signal the workflow should stop
DONE = "done"


class SupervisorAgent(BaseAgent):
    """Routing policy:

    1. If no research_notes  → route to researcher.
    2. If no analysis_notes  → route to analyst.
    3. If no final_answer    → route to writer.
    4. Otherwise             → done.

    Hard stops:
    - max_iterations exceeded → AgentExecutionError (or fallback to writer if
      we at least have research_notes).
    - Any agent recorded an error → stop immediately.
    """

    name = "supervisor"

    def __init__(self) -> None:
        self._settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("supervisor.route", {"iteration": state.iteration}):
            next_route = self._decide(state)

        state.record_route(next_route)
        state.add_trace_event("supervisor.route", {"next": next_route, "iteration": state.iteration})
        logger.info("Supervisor → %s (iteration %d)", next_route, state.iteration)
        return state

    # ------------------------------------------------------------------
    # Internal routing logic
    # ------------------------------------------------------------------

    def _decide(self, state: ResearchState) -> str:
        max_iter = self._settings.max_iterations

        # Hard stop: too many iterations
        if state.iteration >= max_iter:
            logger.warning(
                "Max iterations (%d) reached. Forcing stop.", max_iter
            )
            if state.final_answer:
                return DONE
            if state.research_notes:
                # Fallback: skip analyst and go straight to writer
                logger.warning("Fallback: skipping analyst, routing to writer.")
                return AgentName.WRITER
            raise AgentExecutionError(
                f"Max iterations ({max_iter}) reached without producing any output."
            )

        # Stop if a previous agent recorded an error
        if state.errors:
            logger.error("Stopping due to recorded errors: %s", state.errors)
            return DONE

        # Normal routing
        if state.research_notes is None:
            return AgentName.RESEARCHER

        if state.analysis_notes is None:
            return AgentName.ANALYST

        if state.final_answer is None:
            return AgentName.WRITER

        return DONE
