"""Layer 2 — Retriever: 3-tier cache with async Polars queries.

Tier 1: Response cache hit   →  0 ms  (skips all computation)
Tier 2: Prefetch hit          →  0 ms  (data computed during extraction)
Tier 3: Fresh Polars query    →  5–50 ms  (run off event loop via asyncio.to_thread)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import polars as pl

from src.cache import cache_get, cache_set, make_cache_key
from src.data.loader import get_dataframe

logger = logging.getLogger(__name__)

_PREFETCH_KEY_MAP: dict[str, list[str]] = {
    "pl_summary:2024": ["pl_by_property_2024"],
    "pl_summary:2025": ["pl_by_property_2025"],
    "pl_summary:": ["pl_all"],
    "tenant_query:": ["by_tenant"],
    "comparison:2024": ["comparison_2024"],
    "comparison:2025": ["comparison_2025"],
    "comparison:": ["comparison_2024"],
}


def _prefetch_lookup_key(intent: str, filters: dict) -> str:
    year = str(filters.get("year") or "")
    return f"{intent}:{year}"


async def retriever(state: dict) -> dict:
    """3-tier async retrieval: cache → prefetch → fresh Polars."""
    t0 = time.perf_counter()
    intent: str = state.get("intent") or "general"
    filters: dict[str, Any] = state.get("filters") or {}
    prefetch: dict = state.get("prefetch_results") or {}

    cache_key = make_cache_key(intent, filters)

    # ── Tier 1: Response cache ─────────────────────────────────────────
    cached = cache_get(cache_key)
    if cached is not None:
        return {
            "raw_results": cached,
            "retrieval_error": None,
            "cache_hit": True,
            "cache_tier": "response_cache",
            "timings": {"retriever": time.perf_counter() - t0},
        }

    # ── Tier 2: Prefetch hit ───────────────────────────────────────────
    pkey = _prefetch_lookup_key(intent, filters)
    prefetch_keys = _PREFETCH_KEY_MAP.get(pkey) or _PREFETCH_KEY_MAP.get(f"{intent}:")
    if prefetch_keys:
        for pk in prefetch_keys:
            if pk in prefetch and prefetch[pk]:
                results = prefetch[pk]
                cache_set(cache_key, results)
                logger.debug("Retriever: prefetch hit '%s'", pk)
                return {
                    "raw_results": results,
                    "retrieval_error": None,
                    "cache_hit": False,
                    "cache_tier": "prefetch",
                    "timings": {"retriever": time.perf_counter() - t0},
                }

    # ── Tier 3: Fresh Polars query ─────────────────────────────────────
    try:
        results = await asyncio.to_thread(_run_polars_query, intent, filters)
        cache_set(cache_key, results)
        return {
            "raw_results": results,
            "retrieval_error": None,
            "cache_hit": False,
            "cache_tier": "polars",
            "timings": {"retriever": time.perf_counter() - t0},
        }
    except Exception as exc:
        logger.exception("Retrieval error: %s", exc)
        return {
            "raw_results": [],
            "retrieval_error": str(exc),
            "cache_hit": False,
            "cache_tier": None,
            "timings": {"retriever": time.perf_counter() - t0},
        }


# ── Polars query engine ──────────────────────────────────────────────────


def _run_polars_query(intent: str, filters: dict[str, Any]) -> list[dict]:
    """Synchronous Polars query — called via asyncio.to_thread."""
    df = get_dataframe()
    filtered = _apply_filters(df, filters)
    result = _route_intent(filtered, intent, filters)
    return result.to_dicts() if isinstance(result, pl.DataFrame) else []


def _normalize_ledger_value(val: str, known: set[str]) -> str:
    normalized = val.lower().replace(" ", "_").replace("-", "_")
    if normalized in known:
        return normalized
    from difflib import get_close_matches

    matches = get_close_matches(normalized, list(known), n=1, cutoff=0.6)
    return matches[0] if matches else normalized


def _apply_filters(df: pl.DataFrame, filters: dict[str, Any]) -> pl.DataFrame:
    """Apply all extracted filters to the DataFrame."""
    from src.data.loader import get_cache

    col_map = {
        "property_name": "property_name",
        "tenant_name": "tenant_name",
        "year": "year",
        "quarter": "quarter",
        "month": "month",
        "ledger_group": "ledger_group",
        "ledger_category": "ledger_category",
        "ledger_type": "ledger_type",
    }
    ledger_normalize = {
        "ledger_category": get_cache("available_ledger_categories") or set(),
        "ledger_group": get_cache("available_ledger_groups") or set(),
    }
    mask = pl.lit(True)
    for key, col in col_map.items():
        val = filters.get(key)
        if val and col in df.columns:
            if isinstance(val, list):
                if key in ledger_normalize:
                    val = [_normalize_ledger_value(str(v), ledger_normalize[key]) for v in val]
                mask = mask & pl.col(col).is_in([str(v) for v in val])
            else:
                if key in ledger_normalize:
                    val = _normalize_ledger_value(str(val), ledger_normalize[key])
                mask = mask & (pl.col(col) == str(val))
    return df.filter(mask)


def _route_intent(df: pl.DataFrame, intent: str, filters: dict) -> pl.DataFrame:
    """Route to the appropriate Polars aggregation based on intent."""
    if intent == "pl_summary":
        if filters.get("property_name"):
            return (
                df.group_by("ledger_group")
                .agg(pl.col("profit").sum().alias("total_profit"))
                .sort("total_profit", descending=True)
            )
        return (
            df.filter(pl.col("property_name").is_not_null())
            .group_by("property_name")
            .agg(pl.col("profit").sum().alias("total_profit"))
            .sort("total_profit", descending=True)
        )

    if intent == "property_detail":
        return (
            df.filter(pl.col("property_name").is_not_null())
            .group_by(["property_name", "ledger_group"])
            .agg(pl.col("profit").sum().alias("total_profit"))
            .sort("total_profit", descending=True)
        )

    if intent == "tenant_query":
        return (
            df.filter(pl.col("tenant_name").is_not_null())
            .group_by(["tenant_name", "property_name"])
            .agg(pl.col("profit").sum().alias("total_profit"))
            .sort("total_profit", descending=True)
        )

    if intent == "comparison":
        props = filters.get("property_names") or []
        base = df.filter(pl.col("property_name").is_not_null())
        if props:
            base = base.filter(pl.col("property_name").is_in(props))
        return (
            base.group_by(["property_name", "ledger_type"])
            .agg(pl.col("profit").sum().alias("total_profit"))
            .sort("property_name")
        )

    if intent == "ledger_query":
        return df.select(
            ["ledger_description", "ledger_category", "ledger_group", "month", "profit"]
        ).sort("profit", descending=True)

    # general fallback
    return (
        df.group_by(["year", "ledger_type"])
        .agg(pl.col("profit").sum().alias("total_profit"))
        .sort("year")
    )
