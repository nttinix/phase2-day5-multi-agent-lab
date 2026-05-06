"""Tests for agent implementations.

The original skeleton test verified that SupervisorAgent raised StudentTodoError.
Now that the agents are implemented, we test their actual behaviour using mocks
so no real LLM or search calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.supervisor import DONE
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(query: str = "Explain multi-agent systems") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


def _mock_llm(content: str = "mock response") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        content=content, input_tokens=10, output_tokens=20, cost_usd=0.001
    )
    return llm


def _mock_search(n: int = 2) -> MagicMock:
    search = MagicMock()
    search.search.return_value = [
        SourceDocument(title=f"Source {i}", url=f"https://example.com/{i}", snippet=f"Snippet {i}")
        for i in range(n)
    ]
    return search


# ---------------------------------------------------------------------------
# SupervisorAgent
# ---------------------------------------------------------------------------

class TestSupervisorAgent:
    def test_routes_to_researcher_when_no_notes(self) -> None:
        state = _make_state()
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == AgentName.RESEARCHER

    def test_routes_to_analyst_after_research(self) -> None:
        state = _make_state()
        state.research_notes = "some notes"
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == AgentName.ANALYST

    def test_routes_to_writer_after_analysis(self) -> None:
        state = _make_state()
        state.research_notes = "some notes"
        state.analysis_notes = "some analysis"
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == AgentName.WRITER

    def test_routes_done_when_all_complete(self) -> None:
        state = _make_state()
        state.research_notes = "notes"
        state.analysis_notes = "analysis"
        state.final_answer = "answer"
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == DONE

    def test_stops_on_errors(self) -> None:
        state = _make_state()
        state.errors.append("something went wrong")
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == DONE

    def test_raises_on_max_iterations_without_output(self) -> None:
        state = _make_state()
        # Simulate hitting max_iterations with no output at all
        with patch(
            "multi_agent_research_lab.agents.supervisor.get_settings"
        ) as mock_settings:
            mock_settings.return_value.max_iterations = 0
            with pytest.raises(AgentExecutionError):
                SupervisorAgent().run(state)

    def test_increments_iteration(self) -> None:
        state = _make_state()
        assert state.iteration == 0
        SupervisorAgent().run(state)
        assert state.iteration == 1

    def test_adds_trace_event(self) -> None:
        state = _make_state()
        SupervisorAgent().run(state)
        assert any(e["name"] == "supervisor.route" for e in state.trace)


# ---------------------------------------------------------------------------
# ResearcherAgent
# ---------------------------------------------------------------------------

class TestResearcherAgent:
    def test_populates_sources_and_notes(self) -> None:
        state = _make_state()
        agent = ResearcherAgent(llm_client=_mock_llm("research notes"), search_client=_mock_search())
        result = agent.run(state)
        assert len(result.sources) == 2
        assert result.research_notes == "research notes"

    def test_appends_agent_result(self) -> None:
        state = _make_state()
        agent = ResearcherAgent(llm_client=_mock_llm(), search_client=_mock_search())
        result = agent.run(state)
        assert any(r.agent == AgentName.RESEARCHER for r in result.agent_results)

    def test_adds_trace_event(self) -> None:
        state = _make_state()
        agent = ResearcherAgent(llm_client=_mock_llm(), search_client=_mock_search())
        result = agent.run(state)
        assert any(e["name"] == "researcher.done" for e in result.trace)


# ---------------------------------------------------------------------------
# AnalystAgent
# ---------------------------------------------------------------------------

class TestAnalystAgent:
    def test_populates_analysis_notes(self) -> None:
        state = _make_state()
        state.research_notes = "some research"
        agent = AnalystAgent(llm_client=_mock_llm("analysis output"))
        result = agent.run(state)
        assert result.analysis_notes == "analysis output"

    def test_raises_without_research_notes(self) -> None:
        state = _make_state()
        agent = AnalystAgent(llm_client=_mock_llm())
        with pytest.raises(AgentExecutionError):
            agent.run(state)

    def test_appends_agent_result(self) -> None:
        state = _make_state()
        state.research_notes = "notes"
        agent = AnalystAgent(llm_client=_mock_llm())
        result = agent.run(state)
        assert any(r.agent == AgentName.ANALYST for r in result.agent_results)


# ---------------------------------------------------------------------------
# WriterAgent
# ---------------------------------------------------------------------------

class TestWriterAgent:
    def test_populates_final_answer(self) -> None:
        state = _make_state()
        state.research_notes = "notes"
        state.analysis_notes = "analysis"
        agent = WriterAgent(llm_client=_mock_llm("final answer"))
        result = agent.run(state)
        assert result.final_answer == "final answer"

    def test_raises_without_research_notes(self) -> None:
        state = _make_state()
        agent = WriterAgent(llm_client=_mock_llm())
        with pytest.raises(AgentExecutionError):
            agent.run(state)

    def test_appends_agent_result(self) -> None:
        state = _make_state()
        state.research_notes = "notes"
        agent = WriterAgent(llm_client=_mock_llm())
        result = agent.run(state)
        assert any(r.agent == AgentName.WRITER for r in result.agent_results)


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class TestCriticAgent:
    def test_parses_quality_score(self) -> None:
        state = _make_state()
        state.research_notes = "notes"
        state.final_answer = "answer"
        agent = CriticAgent(llm_client=_mock_llm("SCORE: 8.5\nCRITIQUE: Good answer."))
        result = agent.run(state)
        critic_result = next(r for r in result.agent_results if r.agent == AgentName.CRITIC)
        assert critic_result.metadata["quality_score"] == pytest.approx(8.5)

    def test_skips_when_no_final_answer(self) -> None:
        state = _make_state()
        agent = CriticAgent(llm_client=_mock_llm())
        result = agent.run(state)
        assert not any(r.agent == AgentName.CRITIC for r in result.agent_results)


# ---------------------------------------------------------------------------
# Full workflow (pure-Python loop, no LangGraph required)
# ---------------------------------------------------------------------------

class TestMultiAgentWorkflow:
    def test_end_to_end_loop(self) -> None:
        """Workflow should produce a final_answer without real LLM/search calls."""
        from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow

        llm = _mock_llm("generated content")
        search = _mock_search(3)

        workflow = MultiAgentWorkflow()
        workflow._researcher = ResearcherAgent(llm_client=llm, search_client=search)
        workflow._analyst = AnalystAgent(llm_client=llm)
        workflow._writer = WriterAgent(llm_client=llm)

        state = _make_state()
        result = workflow._run_loop(state)

        assert result.final_answer == "generated content"
        assert result.research_notes == "generated content"
        assert result.analysis_notes == "generated content"
        assert DONE in result.route_history
