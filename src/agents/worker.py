"""Parallel Worker — executes one WorkerAssignment via Polars financial queries.

Invoked N times in parallel via LangGraph Send() fan-out from the planner.
Reuses the retriever's Polars engine (_run_polars_query) so query logic stays DRY.
Each worker appends one WorkerResult to the shared worker_results list
(merged by the operator.add reducer in AgentState).
"""

from __future__ import annotations

import asyncio
import logging
import time

from src.agents.retriever import _run_polars_query

logger = logging.getLogger(__name__)

_WORKER_TYPE_TO_INTENT: dict[str, str] = {
    "pl_worker": "pl_summary",
    "pl_summary": "pl_summary",
    "comparison_worker": "comparison",
    "comparison": "comparison",
    "ledger_worker": "ledger_query",
    "ledger_query": "ledger_query",
    "tenant_worker": "tenant_query",
    "tenant_query": "tenant_query",
    "property_detail": "property_detail",
    "general": "general",
}


async def worker(state: dict) -> dict:
    """Run one Polars financial query for the assigned worker_type + filters."""
    t0 = time.perf_counter()
    assignment = state.get("current_assignment") or {}

    worker_type: str = assignment.get("worker_type") or state.get("intent") or "pl_summary"
    filters: dict = assignment.get("filters") or state.get("filters") or {}
    question: str = assignment.get("question") or state.get("user_query") or ""

    intent = _WORKER_TYPE_TO_INTENT.get(worker_type, "pl_summary")

    try:
        data = await asyncio.to_thread(_run_polars_query, intent, filters)
        error = None
    except Exception as exc:
        logger.exception("Worker %s error: %s", worker_type, exc)
        data = []
        error = str(exc)

    elapsed = time.perf_counter() - t0
    result = {
        "worker_type": worker_type,
        "question": question,
        "filters": filters,
        "data": data,
        "error": error,
    }

    return {
        "worker_results": [result],
        "timings": {f"worker_{worker_type}": elapsed},
    }
