"""Tests for v2 workflow: reason_and_synthesize node and routing."""

from __future__ import annotations

import pytest


_CACHE_STUB = lambda k: {
    "available_properties": {"Building 17", "Building 120"},
    "available_tenants": {"Tenant 4", "Tenant 6"},
    "available_years": ["2024", "2025"],
}.get(k)


# ---------------------------------------------------------------------------
# reason_and_synthesize — critic+judge logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reason_proceeds_with_good_data(monkeypatch):
    monkeypatch.setattr("src.agents.reason_and_synthesize.get_cache", _CACHE_STUB)
    monkeypatch.setattr("src.agents.reason_and_synthesize.chat_complete", __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(return_value="Good data summary."))
    from src.agents.reason_and_synthesize import reason_and_synthesize
    result = await reason_and_synthesize({
        "user_query": "P&L 2024?", "intent": "pl_summary", "filters": {},
        "raw_results": [{"property_name": "Building 17", "total_profit": 100.0}],
        "retrieval_error": None, "iterations": 0, "timings": {},
    })
    assert result["judgment"] is True
    assert result["needs_clarification"] is False
    assert result["final_response"] == "Good data summary."


@pytest.mark.asyncio
async def test_reason_clarifies_on_empty_data(monkeypatch):
    monkeypatch.setattr("src.agents.reason_and_synthesize.get_cache", _CACHE_STUB)
    from src.agents.reason_and_synthesize import reason_and_synthesize
    result = await reason_and_synthesize({
        "user_query": "P&L for Building 999?", "intent": "pl_summary",
        "filters": {"property_name": "Building 999"},
        "raw_results": [], "retrieval_error": None, "iterations": 3, "timings": {},
    })
    assert result["judgment"] is False
    assert result["needs_clarification"] is True
    assert result["clarification_message"] is not None


@pytest.mark.asyncio
async def test_reason_proceeds_with_partial_data_early_iteration(monkeypatch):
    """Even with issues, if data exists and iterations < 2, judge proceeds."""
    monkeypatch.setattr("src.agents.reason_and_synthesize.get_cache", _CACHE_STUB)
    monkeypatch.setattr("src.agents.reason_and_synthesize.chat_complete", __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(return_value="Partial answer."))
    from src.agents.reason_and_synthesize import reason_and_synthesize
    result = await reason_and_synthesize({
        "user_query": "some query", "intent": "general",
        "filters": {"year": "2099"},
        "raw_results": [{"total_profit": 50.0}],
        "retrieval_error": None, "iterations": 0, "timings": {},
    })
    assert result["judgment"] is True


@pytest.mark.asyncio
async def test_reason_handles_list_property(monkeypatch):
    """Critic inside reason_and_synthesize must not crash for list property_name."""
    monkeypatch.setattr("src.agents.reason_and_synthesize.get_cache", _CACHE_STUB)
    monkeypatch.setattr("src.agents.reason_and_synthesize.chat_complete", __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(return_value="Comparison result."))
    from src.agents.reason_and_synthesize import reason_and_synthesize
    result = await reason_and_synthesize({
        "user_query": "Compare 17 vs 120", "intent": "comparison",
        "filters": {"property_name": ["Building 17", "Building 120"], "year": "2024"},
        "raw_results": [{"property_name": "Building 17", "total_profit": 200.0}],
        "retrieval_error": None, "iterations": 0, "timings": {},
    })
    assert result["judgment"] is True
    assert result["critique"] == "OK"


# ---------------------------------------------------------------------------
# Workflow routing
# ---------------------------------------------------------------------------

def test_route_judge_to_end():
    from langgraph.graph import END
    from src.graph.workflow import _route_judge
    assert _route_judge({"needs_clarification": False}) == END


def test_route_judge_to_clarify():
    from src.graph.workflow import _route_judge
    assert _route_judge({"needs_clarification": True}) == "clarify"
