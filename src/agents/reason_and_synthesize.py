"""Layer 3 — Parallel Reasoning + Streaming Synthesis.

Critic+Judge (<1ms Python) and Synthesizer (LLM) start simultaneously via
asyncio.create_task. Synthesis is cancelled cleanly if critic/judge decide
to clarify instead.

Timeline:
  t=0ms:   Critic+Judge task starts  |  Synthesizer task starts
  t=<1ms:  Critic+Judge completes
  t=~150ms: Synthesizer first token arrives (if judgment=True)
  t=~400ms: Synthesis complete
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from src.data.loader import get_cache
from src.llm.local import chat_complete

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM = """\
You are a helpful assistant for PropCo, a real estate asset management company.
When financial data is provided, summarize it clearly using EUR currency formatting (€X,XXX.XX).
Cite the time period when relevant (e.g. "In Q2 2024...").
Never invent financial numbers — only cite numbers from the provided data.
End with a one-sentence summary."""

GENERAL_SYSTEM = """\
You are a helpful assistant. Answer the user's question concisely and accurately
using your general knowledge. If you don't know, say so clearly."""

SYNTHESIS_USER = """\
Query: {user_query}
Intent: {intent}
Filters applied: {filters}
Data:
{raw_results}
Critique notes: {critique}

Response:"""


async def reason_and_synthesize(state: dict) -> dict:
    """Start critic+judge and synthesizer simultaneously; cancel synth if needed."""
    t0 = time.perf_counter()

    critic_task = asyncio.create_task(_run_critic_and_judge(state))
    synth_task = asyncio.create_task(_run_synthesis(state))

    updated = await critic_task

    if not updated.get("judgment", True):
        synth_task.cancel()
        try:
            await synth_task
        except asyncio.CancelledError:
            pass
        elapsed = time.perf_counter() - t0
        timings = updated.get("timings") or {}
        return {**updated, "timings": {**timings, "reason_and_synthesize": elapsed}}

    final_response = await synth_task
    elapsed = time.perf_counter() - t0
    timings = updated.get("timings") or {}
    return {
        **updated,
        "final_response": final_response,
        "timings": {**timings, "reason_and_synthesize": elapsed},
    }


async def _run_critic_and_judge(state: dict) -> dict:
    """Pure Python critic + judge. Completes in < 1 ms."""
    raw_results = _get(state, "raw_results") or []
    filters = _get(state, "filters") or {}
    intent = _get(state, "intent") or "general"
    retrieval_error = _get(state, "retrieval_error")
    iterations = _get(state, "iterations") or 0
    timings = _get(state, "timings") or {}

    # General knowledge queries bypass data validation entirely
    if intent == "general" and not filters and not raw_results:
        return {
            "critique": "OK",
            "judgment": True,
            "needs_clarification": False,
            "iterations": (iterations or 0) + 1,
            "timings": timings,
        }

    issues: list[str] = []

    if retrieval_error:
        issues.append(f"Retrieval error: {retrieval_error}")

    if not raw_results:
        issues.append("No data returned for the given filters.")

    known_props = get_cache("available_properties") or set()
    prop = filters.get("property_name")
    if prop:
        props = prop if isinstance(prop, list) else [prop]
        unknown = [p for p in props if p not in known_props]
        if unknown:
            issues.append(f"Property not found: {unknown}. Available: {sorted(known_props)}")

    known_tenants = get_cache("available_tenants") or set()
    tenant = filters.get("tenant_name")
    if tenant:
        tenants = tenant if isinstance(tenant, list) else [tenant]
        unknown = [t for t in tenants if t not in known_tenants]
        if unknown:
            issues.append(f"Tenant not found: {unknown}. Available: {sorted(known_tenants)}")

    year = filters.get("year")
    if year and str(year) not in (get_cache("available_years") or []):
        issues.append(f"Year '{year}' not in dataset. Available: {get_cache('available_years')}")

    known_cats = get_cache("available_ledger_categories") or set()
    ledger_cat = filters.get("ledger_category")
    if ledger_cat and ledger_cat not in known_cats:
        issues.append(f"Ledger category '{ledger_cat}' not found. Available: {sorted(known_cats)}")

    critique = "; ".join(issues) if issues else "OK"
    proceed = critique == "OK" or (bool(raw_results) and iterations < 2)

    result: dict[str, Any] = {
        "critique": critique,
        "judgment": proceed,
        "needs_clarification": not proceed,
        "iterations": iterations + 1,
        "timings": timings,
    }

    if not proceed:
        if filters.get("ledger_category") or filters.get("ledger_group"):
            hint = f"Known ledger categories: {sorted(known_cats)}."
        elif filters.get("tenant_name"):
            hint = f"Available tenants: {sorted(get_cache('available_tenants') or [])}." 
        else:
            hint = f"Available properties: {sorted(known_props)}."
        result["clarification_message"] = f"Couldn't find the requested data. {critique} {hint}"
        result["final_response"] = result["clarification_message"]

    return result


async def _run_synthesis(state: dict) -> str:
    """LLM synthesis — routes to general knowledge path when no financial data is available."""
    user_query = _get(state, "user_query") or ""
    intent = _get(state, "intent") or "general"
    filters = _get(state, "filters") or {}
    raw_results = _get(state, "raw_results") or []
    critique = _get(state, "critique") or "OK"

    # General knowledge path — no financial data context, let LLM answer freely
    if intent == "general" and not raw_results and not filters:
        return await chat_complete(
            system=GENERAL_SYSTEM,
            user=user_query,
            max_tokens=600,
            temperature=0.3,
        )

    data_str = json.dumps(raw_results[:30], default=str, indent=2) if raw_results else "No data retrieved."

    return await chat_complete(
        system=SYNTHESIS_SYSTEM,
        user=SYNTHESIS_USER.format(
            user_query=user_query,
            intent=intent,
            filters=json.dumps(filters, default=str),
            raw_results=data_str,
            critique=critique,
        ),
        max_tokens=600,
        temperature=0.2,
    )


def _get(state: dict | Any, key: str) -> Any:
    """Safely read from dict or Pydantic model."""
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)
