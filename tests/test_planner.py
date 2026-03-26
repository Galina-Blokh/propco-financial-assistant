"""Tests for Layer 3 — reason_and_synthesize node."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_synthesizer_returns_response(monkeypatch):
    """reason_and_synthesize returns a non-empty final_response given data."""
    from unittest.mock import AsyncMock
    mock = AsyncMock(return_value="Building 17 had a profit of €234,567.00 in 2024.")
    monkeypatch.setattr("src.agents.reason_and_synthesize.chat_complete", mock)

    from src.agents.reason_and_synthesize import reason_and_synthesize
    result = await reason_and_synthesize({
        "user_query": "What is Building 17's profit in 2024?",
        "intent": "pl_summary",
        "filters": {"property_name": "Building 17", "year": "2024"},
        "raw_results": [{"property_name": "Building 17", "total_profit": 234567.0}],
        "critique": "OK",
        "iterations": 0,
    })
    assert result["final_response"] == "Building 17 had a profit of €234,567.00 in 2024."


@pytest.mark.asyncio
async def test_synthesizer_handles_empty_data(monkeypatch):
    """reason_and_synthesize produces a response even with empty results (general path)."""
    from unittest.mock import AsyncMock
    mock = AsyncMock(return_value="No data was found for the given query.")
    monkeypatch.setattr("src.agents.reason_and_synthesize.chat_complete", mock)

    from src.agents.reason_and_synthesize import reason_and_synthesize
    result = await reason_and_synthesize({
        "user_query": "Show Building 999",
        "intent": "general",
        "filters": {},
        "raw_results": [],
        "critique": "OK",
        "iterations": 0,
    })
    assert result["final_response"] is not None
    assert len(result["final_response"]) > 0


@pytest.mark.asyncio
async def test_synthesizer_passes_critique_to_llm(monkeypatch):
    """Critique notes are forwarded into the LLM prompt."""
    from unittest.mock import AsyncMock
    captured_user = {}

    async def mock_complete(system, user, **kwargs):
        captured_user["user"] = user
        return "response"

    monkeypatch.setattr("src.agents.reason_and_synthesize.chat_complete", mock_complete)

    from src.agents.reason_and_synthesize import reason_and_synthesize
    await reason_and_synthesize({
        "user_query": "test",
        "intent": "pl_summary",
        "filters": {"year": "2024"},
        "raw_results": [{"profit": 100}],
        "critique": "Partial data only.",
        "iterations": 0,
    })
    assert "Partial data only." in captured_user.get("user", "")
