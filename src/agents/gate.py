"""Gate node — joins Critic and Synthesizer fan-out (Layer 3 barrier).

Reads the critic verdict.  If OK the synthesized response passes through.
On hard failure (no usable data) it substitutes a clarification message.
On soft failure (partial data with issues) it appends a caveat.
"""

from __future__ import annotations

import logging

from src.data.loader import get_cache

logger = logging.getLogger(__name__)


async def gate(state: dict) -> dict:
    """Decide: pass synthesis through, append caveat, or replace with clarification."""
    critique_ok = state.get("critique_ok", True)
    raw_results = state.get("raw_results") or []
    iterations = state.get("iterations", 0)

    hard_fail = not critique_ok and not raw_results and iterations < 2

    if hard_fail:
        msg = _build_clarification(state)
        return {
            "needs_clarification": True,
            "clarification_message": msg,
            "final_response": msg,
            "iterations": iterations + 1,
        }

    if not critique_ok and raw_results:
        issues = state.get("critique_issues", [])
        caveat = f"\n\n⚠️ Note: {'; '.join(issues)}"
        response = (state.get("final_response") or "") + caveat
        return {
            "needs_clarification": False,
            "final_response": response,
            "iterations": iterations + 1,
        }

    return {
        "needs_clarification": False,
        "iterations": iterations + 1,
    }


def _build_clarification(state: dict) -> str:
    """Build a helpful clarification message.

    Prefers the extractor's LLM-generated clarification question when
    available; falls back to a generic message built from critic issues.
    """
    extractor_question = state.get("clarification_message")
    if extractor_question:
        return extractor_question

    issues = state.get("critique_issues", [])
    filters = state.get("filters") or {}

    lines = ["I couldn't find the requested data."]
    for issue in issues:
        lines.append(f"• {issue}")

    if filters.get("ledger_category") or filters.get("ledger_group"):
        known_cats = sorted(get_cache("available_ledger_categories") or [])
        lines.append(f"\nKnown ledger categories: {known_cats}")
    elif filters.get("tenant_name"):
        known_tenants = sorted(get_cache("available_tenants") or [])
        lines.append(f"\nAvailable tenants: {known_tenants}")
    else:
        known_props = sorted(get_cache("available_properties") or [])
        lines.append(f"\nAvailable properties: {known_props}")

    return "\n".join(lines)
