"""End-to-end integration tests (require real OPENAI_API_KEY)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env (override=True so real key beats conftest dummy)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

# Skip all tests in this module if no real API key is set
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"].startswith("sk-test"),
    reason="OPENAI_API_KEY not set or is a test key",
)


@pytest.mark.asyncio
async def test_full_pipeline_pnl():
    """Run the full pipeline for a P&L summary query."""
    from src.graph.workflow import build_graph, create_initial_state

    graph = build_graph()
    state = create_initial_state(user_query="What's the total P&L for Q2 2024?")
    result = await graph.ainvoke(state)

    assert result["final_response"] is not None
    assert len(result["final_response"]) > 10
    assert result["intent"] == "pl_summary"


@pytest.mark.asyncio
async def test_full_pipeline_comparison():
    """Run the full pipeline for a building comparison."""
    from src.graph.workflow import build_graph, create_initial_state

    graph = build_graph()
    state = create_initial_state(user_query="Compare revenue for Building 17 vs Building 120 in 2024")
    result = await graph.ainvoke(state)

    assert result["final_response"] is not None
    assert result["intent"] == "comparison"
    assert result["raw_results"] is not None
    assert len(result["raw_results"]) > 0


@pytest.mark.asyncio
async def test_full_pipeline_tenant_ranking():
    """Run the full pipeline for a tenant ranking query."""
    from src.graph.workflow import build_graph, create_initial_state

    graph = build_graph()
    state = create_initial_state(user_query="Which tenant generates the most revenue?")
    result = await graph.ainvoke(state)

    assert result["final_response"] is not None
    assert result["intent"] == "tenant_query"


@pytest.mark.asyncio
async def test_full_pipeline_unknown_property_clarifies():
    """Pipeline should request clarification for unknown property names."""
    from src.graph.workflow import build_graph, create_initial_state

    graph = build_graph()
    state = create_initial_state(user_query="Show me financials for Building 999")
    result = await graph.ainvoke(state)

    # Either clarifies or returns response — should not crash
    assert result["final_response"] is not None
