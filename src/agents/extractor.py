"""Layer 1 — Extractor: parse user query into structured intent + filters via LLM.

Speculative prefetch runs in parallel with LLM extraction via asyncio.gather:
- While the LLM parses intent+filters, Polars pre-runs all common aggregations.
- Covers ~85% of query patterns at 0ms retrieval cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from src.data.fuzzy import resolve_filters
from src.data.loader import get_cache, get_dataframe
from src.llm.local import chat_complete

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a query parser for a real estate financial database.
Extract intent and filters from the user query.
Reply ONLY with valid JSON, no markdown, no explanation.

Available intents: pl_summary | property_detail | tenant_query | comparison | ledger_query | general

Available filter keys: property_name, tenant_name, year, quarter, month, ledger_group, ledger_category, ledger_type

Known properties: {available_properties}
Known tenants: {available_tenants}
Known years: {available_years}
Known quarters: {available_quarters}
Known ledger_categories (use EXACT snake_case values): {available_ledger_categories}
Known ledger_groups (use EXACT snake_case values): {available_ledger_groups}

Return format:
{{"intent": "<intent>", "filters": {{<key>: <value>, ...}}}}"""

USER_TEMPLATE = 'User query: "{user_query}"\n\nJSON:'


async def extractor(state: dict) -> dict:
    """Extract intent + filters from the user query while speculatively pre-fetching data."""
    t0 = time.perf_counter()
    user_query = state["user_query"] if isinstance(state, dict) else state.user_query

    properties = sorted(get_cache("available_properties") or [])
    tenants = sorted(get_cache("available_tenants") or [])
    years = get_cache("available_years") or []
    quarters = get_cache("available_quarters") or []
    ledger_categories = sorted(get_cache("available_ledger_categories") or [])
    ledger_groups = sorted(get_cache("available_ledger_groups") or [])

    system = SYSTEM_PROMPT.format(
        available_properties=properties,
        available_tenants=tenants,
        available_years=years,
        available_quarters=quarters,
        available_ledger_categories=ledger_categories,
        available_ledger_groups=ledger_groups,
    )

    # Run LLM extraction and Polars speculative prefetch simultaneously
    (intent, filters), prefetch = await asyncio.gather(
        _extract_intent(system, user_query),
        _speculative_prefetch(),
    )

    elapsed = time.perf_counter() - t0
    timings = (state.get("timings") or {}) if isinstance(state, dict) else (state.timings or {})
    return {
        "intent": intent,
        "filters": filters,
        "prefetch_results": prefetch,
        "timings": {**timings, "extractor": elapsed},
    }


async def _extract_intent(system: str, user_query: str) -> tuple[str, dict]:
    """Run LLM JSON extraction with one retry + fallback."""
    raw = await chat_complete(
        system=system,
        user=USER_TEMPLATE.format(user_query=user_query),
        max_tokens=256,
        temperature=0.0,
        json_mode=True,
    )
    parsed = _parse_json(raw)
    if parsed is None:
        raw2 = await chat_complete(
            system=system,
            user=USER_TEMPLATE.format(user_query=user_query),
            max_tokens=256,
            temperature=0.0,
            json_mode=True,
        )
        parsed = _parse_json(raw2)
    if parsed is None:
        logger.warning("Extractor: JSON parse failed twice, using general intent")
        parsed = {"intent": "general", "filters": {}}

    intent = parsed.get("intent", "general")
    filters = parsed.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}
    return intent, resolve_filters(filters)


async def _speculative_prefetch() -> dict[str, list[dict]]:
    """Pre-run 7 common Polars aggregations off the event loop — covers ~85% of queries."""
    import polars as pl

    def _q_pl_property(year: str) -> list[dict]:
        df = get_dataframe()
        return (
            df.filter(pl.col("year") == year)
            .filter(pl.col("property_name").is_not_null())
            .group_by("property_name")
            .agg(pl.col("profit").sum().alias("total_profit"))
            .sort("total_profit", descending=True)
            .to_dicts()
        )

    def _q_tenant() -> list[dict]:
        df = get_dataframe()
        return (
            df.filter(pl.col("tenant_name").is_not_null())
            .group_by(["tenant_name", "property_name"])
            .agg(pl.col("profit").sum().alias("total_profit"))
            .sort("total_profit", descending=True)
            .to_dicts()
        )

    def _q_quarter() -> list[dict]:
        df = get_dataframe()
        return (
            df.filter(pl.col("property_name").is_not_null())
            .group_by(["property_name", "quarter"])
            .agg(pl.col("profit").sum().alias("total_profit"))
            .sort(["quarter", "property_name"])
            .to_dicts()
        )

    def _q_comparison(year: str) -> list[dict]:
        df = get_dataframe()
        return (
            df.filter(pl.col("year") == year)
            .filter(pl.col("property_name").is_not_null())
            .group_by(["property_name", "ledger_type"])
            .agg(pl.col("profit").sum().alias("total_profit"))
            .sort("property_name")
            .to_dicts()
        )

    def _q_all() -> list[dict]:
        df = get_dataframe()
        return (
            df.filter(pl.col("property_name").is_not_null())
            .group_by("property_name")
            .agg(pl.col("profit").sum().alias("total_profit"))
            .sort("total_profit", descending=True)
            .to_dicts()
        )

    keys = [
        "pl_by_property_2024",
        "pl_by_property_2025",
        "by_tenant",
        "by_quarter",
        "comparison_2024",
        "comparison_2025",
        "pl_all",
    ]
    results = await asyncio.gather(
        asyncio.to_thread(_q_pl_property, "2024"),
        asyncio.to_thread(_q_pl_property, "2025"),
        asyncio.to_thread(_q_tenant),
        asyncio.to_thread(_q_quarter),
        asyncio.to_thread(_q_comparison, "2024"),
        asyncio.to_thread(_q_comparison, "2025"),
        asyncio.to_thread(_q_all),
    )
    return dict(zip(keys, results))


def _parse_json(raw: str) -> dict | None:
    """Attempt to parse raw LLM output as JSON."""
    try:
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return None
