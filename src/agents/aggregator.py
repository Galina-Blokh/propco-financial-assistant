"""Layer 2 — Aggregator: merges parallel worker results into raw_results.

Acts as the synchronization barrier after the Send() fan-out: LangGraph waits
for all worker nodes to complete before executing this node.
Flattens all WorkerResult.data lists into a single raw_results list consumed
by the critic and synthesizer in the next fan-out.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


async def aggregator(state: dict) -> dict:
    """Flatten worker_results[].data into raw_results."""
    t0 = time.perf_counter()
    worker_results: list[dict] = state.get("worker_results") or []

    raw_results: list[dict] = []
    errors: list[str] = []

    for wr in worker_results:
        data = wr.get("data") or []
        raw_results.extend(data)
        if wr.get("error"):
            errors.append(f"{wr.get('worker_type', '?')}: {wr['error']}")

    logger.debug(
        "Aggregator: %d workers → %d rows, %d errors",
        len(worker_results),
        len(raw_results),
        len(errors),
    )

    return {
        "raw_results": raw_results,
        "retrieval_error": "; ".join(errors) if errors else None,
        "cache_hit": False,
        "cache_tier": "worker",
        "timings": {"aggregator": time.perf_counter() - t0},
    }
