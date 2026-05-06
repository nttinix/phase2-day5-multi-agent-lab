"""LangGraph multi-agent workflow.

Graph topology:
  START → supervisor → [researcher | analyst | writer | END]
  Each worker returns to supervisor after completing its step.
"""

from __future__ import annotations

import logging
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


def _state_to_dict(state: ResearchState) -> dict[str, Any]:
    return state.model_dump()


def _dict_to_state(data: dict[str, Any]) -> ResearchState:
    return ResearchState.model_validate(data)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Uses LangGraph when available; falls back to a pure-Python loop so the
    workflow works even without the optional [llm] extras installed.
    """

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()

    # ------------------------------------------------------------------
    # LangGraph graph construction
    # ------------------------------------------------------------------

    def build(self) -> object:
        """Create and return a compiled LangGraph StateGraph."""

        try:
            from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "langgraph is required for MultiAgentWorkflow.build(). "
                "Install it with: pip install 'multi-agent-research-lab[llm]'"
            ) from exc

        # LangGraph nodes receive/return plain dicts
        def supervisor_node(data: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(data)
            state = self._supervisor.run(state)
            return _state_to_dict(state)

        def researcher_node(data: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(data)
            state = self._researcher.run(state)
            return _state_to_dict(state)

        def analyst_node(data: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(data)
            state = self._analyst.run(state)
            return _state_to_dict(state)

        def writer_node(data: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(data)
            state = self._writer.run(state)
            return _state_to_dict(state)

        def route_after_supervisor(data: dict[str, Any]) -> str:
            """Conditional edge: read the last route and return the next node name."""
            route_history: list[str] = data.get("route_history", [])
            if not route_history:
                return AgentName.RESEARCHER
            last = route_history[-1]
            if last == DONE:
                return END
            return last  # "researcher" | "analyst" | "writer"

        graph: StateGraph = StateGraph(dict)
        graph.add_node("supervisor", supervisor_node)
        graph.add_node(AgentName.RESEARCHER, researcher_node)
        graph.add_node(AgentName.ANALYST, analyst_node)
        graph.add_node(AgentName.WRITER, writer_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {
                AgentName.RESEARCHER: AgentName.RESEARCHER,
                AgentName.ANALYST: AgentName.ANALYST,
                AgentName.WRITER: AgentName.WRITER,
                END: END,
            },
        )
        # After each worker, return to supervisor
        graph.add_edge(AgentName.RESEARCHER, "supervisor")
        graph.add_edge(AgentName.ANALYST, "supervisor")
        graph.add_edge(AgentName.WRITER, "supervisor")

        return graph.compile()

    # ------------------------------------------------------------------
    # Run (with LangGraph or pure-Python fallback)
    # ------------------------------------------------------------------

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow and return the final ResearchState."""

        try:
            return self._run_langgraph(state)
        except ImportError:
            logger.warning("langgraph not installed – using pure-Python loop fallback.")
            return self._run_loop(state)

    def _run_langgraph(self, state: ResearchState) -> ResearchState:
        compiled = self.build()
        with trace_span("workflow.langgraph", {"query": state.request.query}):
            result: dict[str, Any] = compiled.invoke(_state_to_dict(state))
        return _dict_to_state(result)

    def _run_loop(self, state: ResearchState) -> ResearchState:
        """Pure-Python fallback: supervisor → worker → supervisor → … until done."""

        _WORKERS = {
            AgentName.RESEARCHER: self._researcher,
            AgentName.ANALYST: self._analyst,
            AgentName.WRITER: self._writer,
        }

        with trace_span("workflow.loop", {"query": state.request.query}):
            while True:
                state = self._supervisor.run(state)
                last_route = state.route_history[-1] if state.route_history else DONE

                if last_route == DONE:
                    logger.info("Workflow complete after %d iterations.", state.iteration)
                    break

                worker = _WORKERS.get(last_route)
                if worker is None:
                    raise AgentExecutionError(f"Unknown route: {last_route!r}")

                try:
                    state = worker.run(state)
                except Exception as exc:  # noqa: BLE001
                    error_msg = f"{last_route} failed: {exc}"
                    logger.error(error_msg)
                    state.errors.append(error_msg)
                    # Let supervisor decide whether to stop or fallback
                    continue

        return state
