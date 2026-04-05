"""Tests for v3 workflow: fan-out topology and gate routing."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


_CACHE_STUB = lambda k: {
    "available_properties": {"Building 17", "Building 120"},
    "available_tenants": {"Tenant 4", "Tenant 6"},
    "available_years": ["2024", "2025"],
    "available_ledger_categories": set(),
}.get(k)


# ---------------------------------------------------------------------------
# Critic — deterministic validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critic_passes_with_good_data(monkeypatch):
    monkeypatch.setattr("src.agents.critic.get_cache", _CACHE_STUB)
    from src.agents.critic import critic

    result = await critic(
        {
            "intent": "pl_summary",
            "filters": {},
            "raw_results": [{"property_name": "Building 17", "total_profit": 100.0}],
            "retrieval_error": None,
        }
    )
    assert result["critique_ok"] is True
    assert result["critique_issues"] == []


@pytest.mark.asyncio
async def test_critic_fails_on_empty_data(monkeypatch):
    monkeypatch.setattr("src.agents.critic.get_cache", _CACHE_STUB)
    from src.agents.critic import critic

    result = await critic(
        {
            "intent": "pl_summary",
            "filters": {"property_name": "Building 999"},
            "raw_results": [],
            "retrieval_error": None,
        }
    )
    assert result["critique_ok"] is False
    assert len(result["critique_issues"]) >= 1


@pytest.mark.asyncio
async def test_critic_bypasses_general_intent(monkeypatch):
    monkeypatch.setattr("src.agents.critic.get_cache", _CACHE_STUB)
    from src.agents.critic import critic

    result = await critic(
        {"intent": "general", "filters": {}, "raw_results": [], "retrieval_error": None}
    )
    assert result["critique_ok"] is True


# ---------------------------------------------------------------------------
# Gate — conditional routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_passes_on_good_critique():
    from src.agents.gate import gate

    result = await gate(
        {
            "critique_ok": True,
            "raw_results": [{"x": 1}],
            "iterations": 0,
            "final_response": "answer",
        }
    )
    assert result["needs_clarification"] is False


@pytest.mark.asyncio
async def test_gate_hard_fail_sets_clarification(monkeypatch):
    monkeypatch.setattr(
        "src.agents.gate.get_cache",
        lambda k: {"available_properties": {"Building 17"}}.get(k, set()),
    )
    from src.agents.gate import gate

    result = await gate(
        {
            "critique_ok": False,
            "critique_issues": ["No data returned"],
            "raw_results": [],
            "iterations": 0,
            "filters": {},
        }
    )
    assert result["needs_clarification"] is True
    assert "couldn't find" in result["final_response"].lower()


@pytest.mark.asyncio
async def test_gate_soft_fail_appends_caveat():
    from src.agents.gate import gate

    result = await gate(
        {
            "critique_ok": False,
            "critique_issues": ["Year '2099' not in dataset"],
            "raw_results": [{"total_profit": 50.0}],
            "iterations": 0,
            "final_response": "Partial answer.",
        }
    )
    assert result["needs_clarification"] is False
    assert "⚠️" in result["final_response"]


# ---------------------------------------------------------------------------
# Workflow routing
# ---------------------------------------------------------------------------


def test_route_gate_to_end():
    from langgraph.graph import END
    from src.graph.workflow import _route_gate

    assert _route_gate({"needs_clarification": False}) == END


def test_route_gate_to_clarify():
    from src.graph.workflow import _route_gate

    assert _route_gate({"needs_clarification": True}) == "clarify"


def test_route_after_join_to_retriever():
    from src.graph.workflow import _route_after_join

    assert _route_after_join({"needs_clarification": False}) == "retriever"
    assert _route_after_join({}) == "retriever"


def test_route_after_join_to_clarify():
    from src.graph.workflow import _route_after_join

    assert _route_after_join({"needs_clarification": True}) == "clarify"


# ---------------------------------------------------------------------------
# Synthesizer — mock LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesizer_returns_response(monkeypatch):
    monkeypatch.setattr(
        "src.agents.synthesizer.chat_complete",
        AsyncMock(return_value="The total P&L for 2024 is €500,000."),
    )
    from src.agents.synthesizer import synthesizer

    result = await synthesizer(
        {
            "user_query": "P&L 2024?",
            "intent": "pl_summary",
            "filters": {"year": "2024"},
            "raw_results": [{"property_name": "Building 17", "total_profit": 500000}],
        }
    )
    assert "€500,000" in result["final_response"]


@pytest.mark.asyncio
async def test_synthesizer_general_knowledge(monkeypatch):
    monkeypatch.setattr(
        "src.agents.synthesizer.chat_complete",
        AsyncMock(return_value="Real estate is property consisting of land."),
    )
    from src.agents.synthesizer import synthesizer

    result = await synthesizer(
        {"user_query": "What is real estate?", "intent": "general", "filters": {}, "raw_results": []}
    )
    assert result["final_response"] is not None
    assert len(result["final_response"]) > 0
