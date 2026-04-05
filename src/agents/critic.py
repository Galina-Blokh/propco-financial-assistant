"""Layer 3a — Critic: deterministic data validation.

Pure Python, no LLM call, completes in < 1 ms.
Runs in parallel with Synthesizer via LangGraph fan-out edges.
"""

from __future__ import annotations

import time

from src.data.loader import get_cache


async def critic(state: dict) -> dict:
    """Validate retrieval results against known entities. No I/O, no LLM."""
    t0 = time.perf_counter()
    raw_results = state.get("raw_results") or []
    filters = state.get("filters") or {}
    intent = state.get("intent") or "general"
    retrieval_error = state.get("retrieval_error")

    # General knowledge queries bypass data validation entirely
    if intent == "general" and not filters and not raw_results:
        return {
            "critique_issues": [],
            "critique_ok": True,
            "timings": {"critic": time.perf_counter() - t0},
        }

    issues: list[str] = []

    if retrieval_error:
        issues.append(f"Retrieval error: {retrieval_error}")

    if not raw_results:
        issues.append("No data returned for the given filters.")

    known_props = get_cache("available_properties") or set()
    prop = filters.get("property_name")
    if prop:
        props_list = prop if isinstance(prop, list) else [prop]
        unknown = [p for p in props_list if p not in known_props]
        if unknown:
            issues.append(f"Property not found: {unknown}. Available: {sorted(known_props)}")

    known_tenants = get_cache("available_tenants") or set()
    tenant = filters.get("tenant_name")
    if tenant:
        tenants_list = tenant if isinstance(tenant, list) else [tenant]
        unknown = [t for t in tenants_list if t not in known_tenants]
        if unknown:
            issues.append(f"Tenant not found: {unknown}. Available: {sorted(known_tenants)}")

    year = filters.get("year")
    if year and str(year) not in (get_cache("available_years") or []):
        issues.append(f"Year '{year}' not in dataset. Available: {get_cache('available_years')}")

    known_cats = get_cache("available_ledger_categories") or set()
    ledger_cat = filters.get("ledger_category")
    if ledger_cat and ledger_cat not in known_cats:
        issues.append(f"Ledger category '{ledger_cat}' not found. Available: {sorted(known_cats)}")

    return {
        "critique_issues": issues,
        "critique_ok": len(issues) == 0,
        "timings": {"critic": time.perf_counter() - t0},
    }
