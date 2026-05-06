"""Tracing hooks.

Khi LANGSMITH_API_KEY được set, mọi span sẽ tự động được gửi lên LangSmith.
Dùng @traceable decorator cho các hàm quan trọng, hoặc trace_span() context manager
cho các block code bất kỳ.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


def _langsmith_enabled() -> bool:
    """Trả về True nếu LANGSMITH_API_KEY được set trong môi trường."""
    return bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager tạo một span.

    - Luôn log duration ra console.
    - Nếu LangSmith được cấu hình, span được gửi lên LangSmith tự động
      thông qua LANGCHAIN_TRACING_V2=true (LangChain auto-instrumentation).

    Usage::

        with trace_span("researcher.search", {"query": q}) as span:
            results = search(q)
            span["attributes"]["num_results"] = len(results)
    """

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": dict(attributes or {}),
        "duration_seconds": None,
    }

    # Nếu LangSmith enabled, tạo run thủ công qua RunTree
    run_tree = _start_run(name, span["attributes"])

    try:
        yield span
    except Exception as exc:
        span["error"] = True
        _end_run(run_tree, span, error=str(exc))
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug(
            "trace_span name=%s duration=%.3fs",
            name,
            span["duration_seconds"],
        )
        if not span.get("error"):
            _end_run(run_tree, span)


# ------------------------------------------------------------------
# LangSmith RunTree integration (best-effort)
# ------------------------------------------------------------------

def _start_run(name: str, inputs: dict[str, Any]) -> Any:
    if not _langsmith_enabled():
        return None
    try:
        from langsmith.run_trees import RunTree  # type: ignore[import-untyped]
        from multi_agent_research_lab.core.config import get_settings

        settings = get_settings()
        run = RunTree(
            name=name,
            run_type="chain",
            inputs=inputs,
            project_name=settings.langsmith_project,
        )
        run.post()
        return run
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith _start_run failed: %s", exc)
        return None


def _end_run(run: Any, span: dict[str, Any], error: str | None = None) -> None:
    if run is None:
        return
    try:
        outputs = {"duration_seconds": span.get("duration_seconds")}
        if error:
            run.end(outputs=outputs, error=error)
        else:
            run.end(outputs=outputs)
        run.patch()
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith _end_run failed: %s", exc)


# ------------------------------------------------------------------
# @traceable decorator helper (wraps langsmith.traceable if available)
# ------------------------------------------------------------------

def traceable(name: str | None = None) -> Any:
    """Decorator that wraps a function with LangSmith tracing when available.

    Falls back to a no-op decorator if langsmith is not installed.

    Usage::

        @traceable("researcher.search")
        def my_function(...):
            ...
    """
    if _langsmith_enabled():
        try:
            from langsmith import traceable as _ls_traceable  # type: ignore[import-untyped]
            return _ls_traceable(name=name)
        except ImportError:
            pass

    # No-op fallback
    def decorator(fn: Any) -> Any:
        return fn

    return decorator
