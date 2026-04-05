"""Retrieval accuracy tests — verify Polars aggregations against real parquet data."""

from __future__ import annotations

import asyncio

from src.agents.retriever import retriever


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_comparison_revenue_and_expenses_per_property():
    """Comparison intent returns both revenue and expenses rows for each property."""
    result = _run(retriever({"intent": "comparison", "filters": {"year": "2024"}, "prefetch_results": {}}))
    rows = result["raw_results"]
    ledger_types = {r["ledger_type"] for r in rows}
    assert "revenue" in ledger_types
    assert "expenses" in ledger_types


def test_pl_summary_profit_is_positive_for_2024():
    from src.data.loader import warm_up

    warm_up()
    result = _run(retriever({"intent": "pl_summary", "filters": {"year": "2024"}, "prefetch_results": {}}))
    rows = result["raw_results"]
    assert len(rows) > 0
    total = sum(r["total_profit"] for r in rows)
    assert total > 0


def test_tenant_query_profit_sum_nonzero():
    result = _run(retriever({"intent": "tenant_query", "filters": {}, "prefetch_results": {}}))
    total = sum(r["total_profit"] for r in result["raw_results"])
    assert total != 0.0


def test_property_detail_groups_by_ledger_group():
    result = _run(
        retriever(
            {"intent": "property_detail", "filters": {"property_name": "Building 140"}, "prefetch_results": {}}
        )
    )
    rows = result["raw_results"]
    groups = {r["ledger_group"] for r in rows}
    assert len(groups) > 1


def test_ledger_query_has_description_column():
    result = _run(
        retriever(
            {"intent": "ledger_query", "filters": {"ledger_group": "bank charges"}, "prefetch_results": {}}
        )
    )
    if result["raw_results"]:
        assert "ledger_description" in result["raw_results"][0]


def test_retrieval_quarter_filter():
    result = _run(
        retriever({"intent": "pl_summary", "filters": {"quarter": "2024-Q2"}, "prefetch_results": {}})
    )
    assert result["retrieval_error"] is None
    assert len(result["raw_results"]) > 0
