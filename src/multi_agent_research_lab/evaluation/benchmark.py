"""Benchmark runner for single-agent vs multi-agent comparison."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def _total_cost(state: ResearchState) -> float | None:
    """Sum cost_usd across all agent results; return None if none recorded."""
    costs = [
        r.metadata["cost_usd"]
        for r in state.agent_results
        if r.metadata.get("cost_usd") is not None
    ]
    return sum(costs) if costs else None


def _citation_coverage(state: ResearchState) -> float | None:
    """Fraction of source documents referenced in the final answer."""
    if not state.final_answer or not state.sources:
        return None
    cited = sum(
        1
        for i in range(len(state.sources))
        if f"[{i + 1}]" in state.final_answer
    )
    return cited / len(state.sources)


def _quality_from_critic(state: ResearchState) -> float | None:
    """Extract quality score left by CriticAgent, if present."""
    for result in state.agent_results:
        score = result.metadata.get("quality_score")
        if score is not None:
            return float(score)
    return None


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    notes: str = "",
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Time *runner*, collect metrics, and return (state, metrics).

    Metrics collected:
    - latency_seconds  – wall-clock time
    - estimated_cost_usd – sum of per-agent LLM costs
    - quality_score    – from CriticAgent if present
    - notes            – citation coverage + any extra notes
    """

    logger.info("Benchmark '%s' starting for query=%r", run_name, query)
    started = perf_counter()

    try:
        state = runner(query)
        error_note = ""
    except Exception as exc:  # noqa: BLE001
        logger.error("Benchmark '%s' failed: %s", run_name, exc)
        from multi_agent_research_lab.core.schemas import ResearchQuery

        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))
        error_note = f" | ERROR: {exc}"

    latency = perf_counter() - started
    cost = _total_cost(state)
    quality = _quality_from_critic(state)
    coverage = _citation_coverage(state)

    coverage_note = (
        f"citation_coverage={coverage:.0%}" if coverage is not None else "citation_coverage=n/a"
    )
    full_notes = f"{coverage_note}{error_note}" + (f" | {notes}" if notes else "")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        notes=full_notes,
    )
    logger.info(
        "Benchmark '%s' done. latency=%.2fs cost=%s quality=%s",
        run_name,
        latency,
        f"${cost:.4f}" if cost else "n/a",
        quality,
    )
    return state, metrics
