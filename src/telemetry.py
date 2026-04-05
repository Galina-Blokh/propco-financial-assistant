"""MLflow telemetry — logs each pipeline query as an MLflow run.

Uses a **single background worker + queue** so logging never spawns many threads,
never holds SQLite locks from the Streamlit thread, and cannot pile up behind
``st.rerun()``. Default tracking URI is a **local file store** under the repo to
avoid SQLite contention with ``mlflow ui`` (see ``config.mlflow_tracking_uri``).
"""

from __future__ import annotations

import logging
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any

import mlflow

from src.utils.config import settings

logger = logging.getLogger(__name__)

_INITIALISED = False
_LOG_QUEUE: "queue.Queue[tuple[dict[str, Any], str] | None]" = queue.Queue()
_WORKER_THREAD: threading.Thread | None = None
_WORKER_LOCK = threading.Lock()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_tracking_uri() -> str:
    """Normalize tracking URI. File URIs pass through unchanged."""
    uri = settings.mlflow_tracking_uri.strip()
    if uri.startswith("file:"):
        return uri
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        tail = uri[len("sqlite:///") :].lstrip("/")
        path = Path(tail)
        if not path.is_absolute():
            path = (_PROJECT_ROOT / path).resolve()
        else:
            path = path.resolve()
        posix = path.as_posix()
        return f"sqlite:///{posix}"
    return uri


def _ensure_init() -> None:
    global _INITIALISED
    if _INITIALISED:
        return
    try:
        resolved = _resolve_tracking_uri()
        mlflow.set_tracking_uri(resolved)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        # Enable LangChain/LangGraph tracing so runs appear in the MLflow Traces tab.
        # Requires MLflow >= 2.12 and langchain installed.  Silently ignored if unavailable.
        try:
            mlflow.langchain.autolog(log_traces=True, silent=True)
            logger.info("MLflow LangChain autolog enabled (Traces tab active)")
        except Exception:
            logger.debug("mlflow.langchain.autolog not available — Traces tab will be empty")
        _INITIALISED = True
        logger.info("MLflow ready: uri=%s experiment=%s", resolved, settings.mlflow_experiment_name)
    except Exception:
        logger.exception("MLflow init failed (non-fatal)")
        _INITIALISED = False


def _log_query_sync(result: dict[str, Any], user_query: str) -> None:
    """Perform one MLflow run (called only from the worker thread)."""
    _ensure_init()
    if not _INITIALISED:
        return

    timings = result.get("timings") or {}
    total_s = sum(timings.values())

    with mlflow.start_run(run_name=_truncate(user_query, 80)):
        mlflow.log_param("user_query", _truncate(user_query, 250))
        mlflow.log_param("intent", result.get("intent") or "unknown")
        mlflow.log_param("cache_tier", result.get("cache_tier") or "none")
        mlflow.log_param("cache_hit", result.get("cache_hit", False))
        mlflow.log_param("critique_ok", result.get("critique_ok", True))
        mlflow.log_param("needs_clarification", result.get("needs_clarification", False))
        mlflow.log_param("llm_backend", settings.llm_backend)

        filters = result.get("filters") or {}
        for k, v in list(filters.items())[:10]:
            mlflow.log_param(f"filter.{k}", _truncate(str(v), 250))

        mlflow.log_metric("total_latency_ms", total_s * 1000)
        for node, secs in timings.items():
            mlflow.log_metric(f"latency_{node}_ms", secs * 1000)

        raw = result.get("raw_results") or []
        mlflow.log_metric("result_row_count", len(raw))
        mlflow.log_metric("iterations", result.get("iterations", 0))

        critique_issues = result.get("critique_issues") or []
        mlflow.log_metric("critique_issue_count", len(critique_issues))

        mlflow.set_tag("pipeline_version", "v3")
        if critique_issues:
            mlflow.set_tag("critique_issues", "; ".join(critique_issues)[:250])

        if settings.mlflow_log_artifacts:
            response = result.get("final_response") or ""
            if response:
                with tempfile.TemporaryDirectory() as td:
                    prompt_path = Path(td) / "prompt.txt"
                    prompt_path.write_text(user_query, encoding="utf-8")
                    response_path = Path(td) / "response.txt"
                    response_path.write_text(response, encoding="utf-8")
                    mlflow.log_artifacts(td, artifact_path="query")

        logger.info(
            "MLflow run logged: experiment=%s query=%s…",
            settings.mlflow_experiment_name,
            user_query[:40],
        )


def _worker_loop() -> None:
    while True:
        item = _LOG_QUEUE.get()
        if item is None:
            return
        payload, qtext = item
        try:
            _log_query_sync(payload, qtext)
        except Exception:
            logger.exception("MLflow worker: log failed")


def _ensure_worker_started() -> None:
    global _WORKER_THREAD
    with _WORKER_LOCK:
        if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
            return
        t = threading.Thread(
            target=_worker_loop,
            daemon=True,
            name="mlflow-telemetry-worker",
        )
        t.start()
        _WORKER_THREAD = t


def enqueue_query_log(result: dict[str, Any], user_query: str) -> None:
    """Queue a log job (non-blocking). Safe to call from the Streamlit thread."""
    if not settings.mlflow_enabled:
        return
    _ensure_worker_started()
    # Shallow copy so caller can mutate session state without affecting the worker
    _LOG_QUEUE.put((dict(result), user_query))


def log_query(result: dict[str, Any], user_query: str) -> None:
    """Backward-compatible name: queues the same as ``enqueue_query_log``."""
    enqueue_query_log(result, user_query)


def get_tracking_uri() -> str:
    """Return the resolved MLflow tracking URI (for display in UI)."""
    return _resolve_tracking_uri()


def _truncate(s: str, max_len: int) -> str:
    return s[:max_len] if len(s) > max_len else s


__all__ = ["enqueue_query_log", "log_query", "get_tracking_uri"]
