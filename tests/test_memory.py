"""Tests for data loader accuracy against known dataset values."""

from __future__ import annotations


def test_profit_by_year_is_positive():
    from src.data.loader import get_cache, warm_up
    warm_up()
    profit = get_cache("total_profit_by_year")
    assert profit["2024"] > 0


def test_profit_by_property_building17_exists():
    from src.data.loader import get_cache, warm_up
    warm_up()
    profit = get_cache("total_profit_by_property")
    assert "Building 17" in profit
    assert isinstance(profit["Building 17"], float)


def test_profit_by_quarter_has_entries():
    from src.data.loader import get_cache, warm_up
    warm_up()
    profit = get_cache("total_profit_by_quarter")
    assert len(profit) >= 4
    assert any(k.startswith("2024") for k in profit)


def test_available_quarters_format():
    from src.data.loader import get_cache, warm_up
    warm_up()
    quarters = get_cache("available_quarters")
    assert all("-Q" in q for q in quarters)


def test_all_five_properties_present():
    from src.data.loader import get_cache, warm_up
    warm_up()
    props = get_cache("available_properties")
    expected = {"Building 17", "Building 120", "Building 140", "Building 160", "Building 180"}
    assert expected.issubset(props)


def test_profit_by_tenant_non_empty():
    from src.data.loader import get_cache, warm_up
    warm_up()
    profit = get_cache("total_profit_by_tenant")
    assert len(profit) > 0
