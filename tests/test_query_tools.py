"""Tests for Layer 2 — Retrieval agent (deterministic Polars query router)."""

from __future__ import annotations

import pytest

from src.agents.retrieval import retrieval_sync as retrieval


def test_retrieval_pl_summary_returns_rows():
    result = retrieval({"intent": "pl_summary", "filters": {}})
    assert result["retrieval_error"] is None
    assert len(result["raw_results"]) > 0
    assert "total_profit" in result["raw_results"][0]


def test_retrieval_pl_summary_with_year_filter():
    result = retrieval({"intent": "pl_summary", "filters": {"year": "2024"}})
    assert result["retrieval_error"] is None
    assert len(result["raw_results"]) > 0


def test_retrieval_comparison_returns_by_ledger_type():
    result = retrieval({"intent": "comparison", "filters": {"year": "2024"}})
    assert result["retrieval_error"] is None
    rows = result["raw_results"]
    assert len(rows) > 0
    assert "ledger_type" in rows[0]
    assert "property_name" in rows[0]


def test_retrieval_comparison_with_property_list():
    result = retrieval({
        "intent": "comparison",
        "filters": {"property_name": ["Building 17", "Building 120"], "year": "2024"},
    })
    assert result["retrieval_error"] is None
    names = {r["property_name"] for r in result["raw_results"]}
    assert names.issubset({"Building 17", "Building 120"})


def test_retrieval_tenant_query_returns_tenants():
    result = retrieval({"intent": "tenant_query", "filters": {}})
    assert result["retrieval_error"] is None
    assert "tenant_name" in result["raw_results"][0]


def test_retrieval_property_detail_includes_ledger_group():
    result = retrieval({
        "intent": "property_detail",
        "filters": {"property_name": "Building 17"},
    })
    assert result["retrieval_error"] is None
    assert "ledger_group" in result["raw_results"][0]


def test_retrieval_general_fallback():
    result = retrieval({"intent": "general", "filters": {}})
    assert result["retrieval_error"] is None
    assert len(result["raw_results"]) > 0


def test_retrieval_unknown_property_returns_empty():
    result = retrieval({
        "intent": "pl_summary",
        "filters": {"property_name": "Building 9999"},
    })
    assert result["retrieval_error"] is None
    assert result["raw_results"] == []
