"""AgentState — shared LangGraph state passed through all nodes.

Parallel nodes write to non-overlapping fields. The ``timings`` dict uses a
custom merge reducer so concurrent timing updates from fan-out nodes are
preserved rather than overwritten.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _merge_dicts(left: dict, right: dict) -> dict:
    """Reducer: merge two dicts (right wins on key collision)."""
    return {**(left or {}), **(right or {})}


class WorkerAssignment(TypedDict, total=False):
    """One unit of work dispatched to a parallel worker via Send()."""

    worker_type: str        # "pl_worker" | "comparison_worker" | "ledger_worker" | etc.
    filters: dict[str, Any]
    question: str           # specific sub-question for this worker


class WorkerResult(TypedDict, total=False):
    """Result returned by one parallel worker."""

    worker_type: str
    question: str
    filters: dict[str, Any]
    data: list[dict[str, Any]]
    error: str | None


class AgentState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────
    user_query: str
    conversation_history: list[dict[str, str]]

    # ── Planner output (Layer 1) ─────────────────────────────────────────
    intent: str
    filters: dict[str, Any]
    worker_assignments: list[WorkerAssignment]
    prefetch_results: dict[str, list[dict]]

    # ── Per-worker context (set via Send()) ──────────────────────────────
    current_assignment: WorkerAssignment

    # ── Parallel worker results — operator.add reducer merges lists ──────
    worker_results: Annotated[list[WorkerResult], operator.add]

    # ── Aggregator output (Layer 2) ──────────────────────────────────────
    raw_results: list[dict] | None
    retrieval_error: str | None
    cache_hit: bool
    cache_tier: str | None  # "response_cache" | "prefetch" | "polars" | "worker"

    # ── Critic output (Layer 3a) ─────────────────────────────────────────
    critique_issues: list[str]
    critique_ok: bool

    # ── Synthesizer output (Layer 3b) ────────────────────────────────────
    # token_queue: asyncio.Queue injected before invocation; never serialised.
    token_queue: Any
    final_response: str | None

    # ── Gate / control flow ──────────────────────────────────────────────
    needs_clarification: bool
    clarification_message: str | None
    iterations: int

    # ── Telemetry ────────────────────────────────────────────────────────
    timings: Annotated[dict[str, float], _merge_dicts]
