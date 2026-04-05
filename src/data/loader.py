"""Pre-loaded in-memory DataFrame and startup caches for zero-latency queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src.utils.config import settings

_DF: pl.DataFrame | None = None
_CACHES: dict[str, Any] = {}


def get_dataframe() -> pl.DataFrame:
    """Return the singleton in-memory DataFrame, loading on first call."""
    global _DF
    if _DF is None:
        path = settings.cortex_data_path
        if not Path(path).exists():
            path = str(Path(__file__).resolve().parents[2] / "cortex.parquet")
        _DF = pl.read_parquet(path)
    return _DF


def get_cache(key: str) -> Any:
    """Return a pre-computed cache value, building all caches on first call."""
    if not _CACHES:
        _build_caches()
    return _CACHES.get(key)


def warm_up() -> None:
    """Pre-load data and build all caches at startup (once per process)."""
    if _CACHES:
        return
    _build_caches()


def _build_caches() -> None:
    """Build all startup aggregation caches from the in-memory DataFrame."""
    df = get_dataframe()

    _CACHES["available_properties"] = set(
        df.filter(pl.col("property_name").is_not_null())
        .select("property_name")
        .unique()
        .to_series()
        .to_list()
    )
    _CACHES["available_tenants"] = set(
        df.filter(pl.col("tenant_name").is_not_null())
        .select("tenant_name")
        .unique()
        .to_series()
        .to_list()
    )
    _CACHES["available_years"] = sorted(
        df.select("year").unique().to_series().to_list()
    )
    _CACHES["available_quarters"] = sorted(
        df.select("quarter").unique().to_series().to_list()
    )
    _CACHES["available_ledger_groups"] = set(
        df.select("ledger_group").unique().to_series().to_list()
    )
    _CACHES["available_ledger_categories"] = set(
        df.select("ledger_category").unique().to_series().to_list()
    )

    year_agg = (
        df.group_by("year")
        .agg(pl.col("profit").sum().alias("profit"))
        .to_dicts()
    )
    _CACHES["total_profit_by_year"] = {r["year"]: r["profit"] for r in year_agg}

    prop_agg = (
        df.filter(pl.col("property_name").is_not_null())
        .group_by("property_name")
        .agg(pl.col("profit").sum().alias("profit"))
        .to_dicts()
    )
    _CACHES["total_profit_by_property"] = {r["property_name"]: r["profit"] for r in prop_agg}

    tenant_agg = (
        df.filter(pl.col("tenant_name").is_not_null())
        .group_by("tenant_name")
        .agg(pl.col("profit").sum().alias("profit"))
        .to_dicts()
    )
    _CACHES["total_profit_by_tenant"] = {r["tenant_name"]: r["profit"] for r in tenant_agg}

    quarter_agg = (
        df.group_by("quarter")
        .agg(pl.col("profit").sum().alias("profit"))
        .to_dicts()
    )
    _CACHES["total_profit_by_quarter"] = {r["quarter"]: r["profit"] for r in quarter_agg}
