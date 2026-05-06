"""Command-line entrypoint for the lab starter."""

from __future__ import annotations

import os

# Kích hoạt LangSmith tracing sớm nhất có thể,
# trước khi bất kỳ LLM client nào được khởi tạo
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


# ------------------------------------------------------------------
# baseline command
# ------------------------------------------------------------------

@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline: one LLM call, no orchestration."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    llm = LLMClient()
    system_prompt = (
        "You are a knowledgeable research assistant. Answer the user's query "
        "thoroughly and accurately in 400-600 words."
    )
    try:
        response = llm.complete(system_prompt, query)
        state.final_answer = response.content
    except Exception as exc:  # noqa: BLE001
        state.final_answer = (
            f"[Baseline could not call LLM: {exc}]\n\n"
            "Set OPENAI_API_KEY in your .env file to enable real completions."
        )

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


# ------------------------------------------------------------------
# multi-agent command
# ------------------------------------------------------------------

@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the full multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    if result.final_answer:
        console.print(Panel.fit(result.final_answer, title="Multi-Agent Answer"))
    else:
        console.print("[yellow]Workflow finished but produced no final_answer.[/yellow]")

    console.print(f"\n[dim]Route history: {' → '.join(result.route_history)}[/dim]")
    console.print(f"[dim]Iterations: {result.iteration}[/dim]")


# ------------------------------------------------------------------
# benchmark command
# ------------------------------------------------------------------

@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")] = (
        "Research GraphRAG state-of-the-art and write a 500-word summary"
    ),
    output: Annotated[str, typer.Option("--output", "-o", help="Report filename")] = (
        "benchmark_report.md"
    ),
) -> None:
    """Benchmark single-agent baseline vs multi-agent workflow and save a report."""

    _init()
    console.print(f"[bold]Benchmarking query:[/bold] {query}\n")

    # --- Baseline runner ---
    def baseline_runner(q: str) -> ResearchState:
        req = ResearchQuery(query=q)
        st = ResearchState(request=req)
        llm = LLMClient()
        system = (
            "You are a knowledgeable research assistant. Answer the user's query "
            "thoroughly and accurately in 400-600 words."
        )
        response = llm.complete(system, q)
        st.final_answer = response.content
        return st

    # --- Multi-agent runner ---
    def multi_agent_runner(q: str) -> ResearchState:
        st = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow().run(st)

    console.print("[cyan]Running baseline…[/cyan]")
    _, baseline_metrics = run_benchmark("single-agent-baseline", query, baseline_runner)

    console.print("[cyan]Running multi-agent workflow…[/cyan]")
    _, multi_metrics = run_benchmark("multi-agent-workflow", query, multi_agent_runner)

    report = render_markdown_report([baseline_metrics, multi_metrics])
    store = LocalArtifactStore()
    path = store.write_text(output, report)

    console.print(Panel.fit(report, title="Benchmark Report"))
    console.print(f"\n[green]Report saved to:[/green] {path}")


if __name__ == "__main__":
    app()
