"""Retrieval accuracy tests — verify Polars aggregations against real parquet data."""

from __future__ import annotations

from src.agents.retrieval import retrieval_sync as retrieval


def test_comparison_revenue_and_expenses_per_property():
    """Comparison intent returns both revenue and expenses rows for each property."""
    result = retrieval({"intent": "comparison", "filters": {"year": "2024"}})
    rows = result["raw_results"]
    ledger_types = {r["ledger_type"] for r in rows}
    assert "revenue" in ledger_types
    assert "expenses" in ledger_types


def test_pl_summary_profit_is_positive_for_2024():
    """pl_summary aggregation for 2024 returns a positive total profit."""
    from src.data.loader import warm_up
    warm_up()

    result = retrieval({"intent": "pl_summary", "filters": {"year": "2024"}})
    rows = result["raw_results"]
    assert len(rows) > 0
    total_from_retrieval = sum(r["total_profit"] for r in rows)
    assert total_from_retrieval > 0


def test_tenant_query_profit_sum_nonzero():
    result = retrieval({"intent": "tenant_query", "filters": {}})
    total = sum(r["total_profit"] for r in result["raw_results"])
    assert total != 0.0


def test_property_detail_groups_by_ledger_group():
    result = retrieval({
        "intent": "property_detail",
        "filters": {"property_name": "Building 140"},
    })
    rows = result["raw_results"]
    groups = {r["ledger_group"] for r in rows}
    assert len(groups) > 1


def test_ledger_query_has_description_column():
    result = retrieval({
        "intent": "ledger_query",
        "filters": {"ledger_group": "bank charges"},
    })
    if result["raw_results"]:
        assert "ledger_description" in result["raw_results"][0]


def test_retrieval_quarter_filter():
    result = retrieval({"intent": "pl_summary", "filters": {"quarter": "2024-Q2"}})
    assert result["retrieval_error"] is None
    assert len(result["raw_results"]) > 0
