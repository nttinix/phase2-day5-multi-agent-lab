"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily when TAVILY_API_KEY is set, otherwise falls back to a
    lightweight mock so the workflow can run without external credentials.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to *query*.

        Tries Tavily first; falls back to mock results if the key is absent
        or the package is not installed.
        """

        if self._settings.tavily_api_key:
            try:
                return self._tavily_search(query, max_results)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tavily search failed (%s) – using mock results.", exc)

        return self._mock_search(query, max_results)

    # ------------------------------------------------------------------
    # Tavily backend
    # ------------------------------------------------------------------

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        from tavily import TavilyClient  # type: ignore[import-untyped]

        client = TavilyClient(api_key=self._settings.tavily_api_key)
        response = client.search(query=query, max_results=max_results)
        results: list[SourceDocument] = []
        for item in response.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score")},
                )
            )
        logger.debug("Tavily returned %d results for query=%r", len(results), query)
        return results

    # ------------------------------------------------------------------
    # Mock backend (no external dependency)
    # ------------------------------------------------------------------

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return plausible-looking stub documents for offline / test use."""

        logger.info("SearchClient using mock results (no TAVILY_API_KEY set).")
        stubs = [
            SourceDocument(
                title=f"Overview of {query}",
                url="https://example.com/overview",
                snippet=(
                    f"This article provides a comprehensive overview of {query}, "
                    "covering key concepts, recent advances, and practical applications."
                ),
            ),
            SourceDocument(
                title=f"Recent advances in {query}",
                url="https://example.com/advances",
                snippet=(
                    f"Researchers have made significant progress in {query} over the past year. "
                    "New benchmarks show improved performance across multiple dimensions."
                ),
            ),
            SourceDocument(
                title=f"Practical guide to {query}",
                url="https://example.com/guide",
                snippet=(
                    f"A step-by-step guide to implementing {query} in production systems, "
                    "including common pitfalls and best practices."
                ),
            ),
            SourceDocument(
                title=f"Limitations and open problems in {query}",
                url="https://example.com/limitations",
                snippet=(
                    f"Despite recent progress, {query} still faces challenges such as "
                    "scalability, interpretability, and data efficiency."
                ),
            ),
            SourceDocument(
                title=f"Case studies: {query} in industry",
                url="https://example.com/case-studies",
                snippet=(
                    f"Several companies have deployed {query} at scale. "
                    "This article examines real-world outcomes and lessons learned."
                ),
            ),
        ]
        return stubs[:max_results]
