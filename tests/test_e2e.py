"""End-to-end integration tests (require LLM API key)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY")
    or os.environ["OPENAI_API_KEY"].startswith("sk-test"),
    reason="OPENAI_API_KEY not set or is a test key",
)


@pytest.mark.asyncio
async def test_full_pipeline_pnl():
    """Run the full v3 pipeline for a P&L query."""
    from src.graph.workflow import build_graph, create_initial_state

    graph = build_graph()
    state = create_initial_state(user_query="What's the total P&L for Q2 2024?")
    result = await graph.ainvoke(state)

    assert result["final_response"] is not None
    assert len(result["final_response"]) > 0
    assert result.get("intent") is not None


@pytest.mark.asyncio
async def test_full_pipeline_comparison():
    """Run the full v3 pipeline for a building comparison."""
    from src.graph.workflow import build_graph, create_initial_state

    graph = build_graph()
    state = create_initial_state(
        user_query="Compare revenue for Building 17 vs Building 120 in 2024"
    )
    result = await graph.ainvoke(state)

    assert result["final_response"] is not None
    assert result.get("intent") in ("comparison", "pl_summary")


@pytest.mark.asyncio
async def test_full_pipeline_general_knowledge():
    """General knowledge queries don't crash and return a response."""
    from src.graph.workflow import build_graph, create_initial_state

    graph = build_graph()
    state = create_initial_state(user_query="What is real estate?")
    result = await graph.ainvoke(state)

    assert result["final_response"] is not None
    assert len(result["final_response"]) > 0
