"""Tests for the data loader and startup caches."""

from __future__ import annotations

import polars as pl


def test_warm_up_loads_dataframe():
    from src.data.loader import get_dataframe, warm_up
    warm_up()
    df = get_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0


def test_warm_up_builds_property_cache():
    from src.data.loader import get_cache, warm_up
    warm_up()
    props = get_cache("available_properties")
    assert isinstance(props, set)
    assert "Building 17" in props


def test_warm_up_builds_year_cache():
    from src.data.loader import get_cache, warm_up
    warm_up()
    years = get_cache("available_years")
    assert "2024" in years


def test_warm_up_builds_profit_by_year():
    from src.data.loader import get_cache, warm_up
    warm_up()
    profit = get_cache("total_profit_by_year")
    assert isinstance(profit, dict)
    assert "2024" in profit
    assert isinstance(profit["2024"], float)


def test_warm_up_builds_profit_by_property():
    from src.data.loader import get_cache, warm_up
    warm_up()
    profit = get_cache("total_profit_by_property")
    assert "Building 17" in profit


def test_warm_up_builds_tenant_cache():
    from src.data.loader import get_cache, warm_up
    warm_up()
    tenants = get_cache("available_tenants")
    assert isinstance(tenants, set)
    assert len(tenants) > 0


def test_warm_up_idempotent():
    """Calling warm_up() twice must not raise or corrupt caches."""
    from src.data.loader import get_cache, warm_up
    warm_up()
    warm_up()
    assert get_cache("available_properties") is not None
