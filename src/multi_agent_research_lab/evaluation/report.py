"""Benchmark report rendering."""

from __future__ import annotations

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a Markdown table with summary analysis."""

    lines = [
        "# Benchmark Report",
        "",
        "## Results",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]

    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |"
        )

    # Summary analysis section
    lines += ["", "## Analysis", ""]

    if len(metrics) >= 2:
        baseline = metrics[0]
        multi = metrics[1]

        latency_delta = multi.latency_seconds - baseline.latency_seconds
        latency_pct = (latency_delta / baseline.latency_seconds * 100) if baseline.latency_seconds else 0
        direction = "slower" if latency_delta > 0 else "faster"
        lines.append(
            f"- **Latency**: Multi-agent is {abs(latency_pct):.1f}% {direction} than baseline "
            f"({baseline.latency_seconds:.2f}s vs {multi.latency_seconds:.2f}s)."
        )

        if baseline.quality_score is not None and multi.quality_score is not None:
            q_delta = multi.quality_score - baseline.quality_score
            lines.append(
                f"- **Quality**: Multi-agent scored {multi.quality_score:.1f}/10 vs "
                f"baseline {baseline.quality_score:.1f}/10 (Δ {q_delta:+.1f})."
            )

        if baseline.estimated_cost_usd is not None and multi.estimated_cost_usd is not None:
            c_delta = multi.estimated_cost_usd - baseline.estimated_cost_usd
            lines.append(
                f"- **Cost**: Multi-agent cost ${multi.estimated_cost_usd:.4f} vs "
                f"baseline ${baseline.estimated_cost_usd:.4f} (Δ ${c_delta:+.4f})."
            )
    else:
        lines.append("_Add a second run to see comparative analysis._")

    lines += [
        "",
        "## When to use multi-agent",
        "",
        "- **Use multi-agent** when the task naturally decomposes into distinct roles "
        "(search, analyse, write), when quality matters more than latency, or when "
        "individual steps benefit from specialised prompts.",
        "- **Avoid multi-agent** for simple, single-step queries where the overhead of "
        "orchestration outweighs the quality gain, or when strict latency/cost budgets apply.",
        "",
    ]

    return "\n".join(lines)
