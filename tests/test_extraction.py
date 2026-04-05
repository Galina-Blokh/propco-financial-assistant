"""Tests for Layer 1 — Extractor agent."""

from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_extractor_comparison(monkeypatch):
    """Extractor returns comparison intent with two properties."""
    from unittest.mock import AsyncMock

    extraction = {
        "intent": "comparison",
        "filters": {"property_name": ["Building 17", "Building 120"], "year": "2024"},
        "confidence": 0.95,
        "needs_clarification": False,
        "clarification_question": None,
    }
    mock = AsyncMock(return_value=json.dumps(extraction))
    monkeypatch.setattr("src.agents.extractor.chat_complete", mock)
    monkeypatch.setattr("src.data.fuzzy.get_cache", lambda k: {"available_properties": {"Building 17", "Building 120"}, "available_tenants": set()}.get(k, set()))

    from src.agents.extractor import extractor
    result = await extractor({"user_query": "Compare Building 17 vs Building 120 in 2024"})

    assert result["intent"] == "comparison"
    assert "Building 17" in result["filters"]["property_name"]
    assert "Building 120" in result["filters"]["property_name"]


@pytest.mark.asyncio
async def test_extractor_fallback_on_bad_json(monkeypatch):
    """Extractor falls back to general intent when LLM returns bad JSON twice."""
    from unittest.mock import AsyncMock

    mock = AsyncMock(return_value="not valid json")
    monkeypatch.setattr("src.agents.extractor.chat_complete", mock)

    from src.agents.extractor import extractor
    result = await extractor({"user_query": "some query"})

    assert result["intent"] == "general"
    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extractor_pl_summary_with_year(monkeypatch):
    """Extractor correctly extracts year filter for P&L summary."""
    from unittest.mock import AsyncMock

    extraction = {
        "intent": "pl_summary",
        "filters": {"year": "2024"},
        "confidence": 0.9,
    }
    mock = AsyncMock(return_value=json.dumps(extraction))
    monkeypatch.setattr("src.agents.extractor.chat_complete", mock)

    from src.agents.extractor import extractor
    result = await extractor({"user_query": "What is the total P&L for 2024?"})

    assert result["intent"] == "pl_summary"
    assert result["filters"]["year"] == "2024"


@pytest.mark.asyncio
async def test_extractor_tenant_query(monkeypatch):
    """Extractor handles tenant queries."""
    from unittest.mock import AsyncMock

    extraction = {
        "intent": "tenant_query",
        "filters": {"ledger_type": "revenue"},
        "confidence": 0.85,
    }
    mock = AsyncMock(return_value=json.dumps(extraction))
    monkeypatch.setattr("src.agents.extractor.chat_complete", mock)

    from src.agents.extractor import extractor
    result = await extractor({"user_query": "Which tenant generates the most revenue?"})

    assert result["intent"] == "tenant_query"


@pytest.mark.asyncio
async def test_extractor_with_conversation_history(monkeypatch):
    """Extractor passes conversation history to the LLM."""
    from unittest.mock import AsyncMock

    extraction = {
        "intent": "pl_summary",
        "filters": {"property_name": "Building 17", "year": "2024"},
        "confidence": 0.9,
    }
    mock = AsyncMock(return_value=json.dumps(extraction))
    monkeypatch.setattr("src.agents.extractor.chat_complete", mock)

    from src.agents.extractor import extractor
    result = await extractor({
        "user_query": "What about last year?",
        "conversation_history": [
            {"role": "user", "content": "Show P&L for Building 17 in 2025"},
            {"role": "assistant", "content": "The P&L for Building 17 in 2025 is..."},
        ],
    })

    assert result["intent"] == "pl_summary"
    call_args = mock.call_args
    assert "Conversation so far" in call_args.kwargs.get("user", call_args[1].get("user", ""))


@pytest.mark.asyncio
async def test_extractor_ambiguity_triggers_clarification(monkeypatch):
    """Extractor signals clarification when query is ambiguous."""
    from unittest.mock import AsyncMock

    extraction = {
        "intent": "pl_summary",
        "filters": {},
        "confidence": 0.3,
        "needs_clarification": True,
        "clarification_question": "Which building or time period are you interested in?",
    }
    mock = AsyncMock(return_value=json.dumps(extraction))
    monkeypatch.setattr("src.agents.extractor.chat_complete", mock)

    from src.agents.extractor import extractor
    result = await extractor({"user_query": "show data"})

    assert result.get("needs_clarification") is True
    assert "which" in result["clarification_message"].lower()
