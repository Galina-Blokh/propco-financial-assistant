"""LangGraph async pipeline — v2 optimized 3-node graph.

Node 1: extractor          — LLM intent+filters  \u2016  speculative Polars prefetch (parallel)
Node 2: retrieval          — 3-tier: cache / prefetch hit / fresh Polars via asyncio.to_thread
Node 3: reason_and_synth   — Critic+Judge (<1ms) \u2016 Synthesizer (LLM) started simultaneously

All nodes are async def. Independent work uses asyncio.gather / asyncio.create_task.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.extractor import extractor
from src.agents.reason_and_synthesize import reason_and_synthesize
from src.agents.retrieval import retrieval
from src.data.loader import warm_up
from src.graph.state import AgentState


# ---------------------------------------------------------------------------
# Routing helper (used by tests)
# ---------------------------------------------------------------------------

def _route_judge(state) -> str:
    """Route after reason_and_synthesize: END normally, clarify if needed."""
    needs = state.get("needs_clarification") if isinstance(state, dict) else state.needs_clarification
    return "clarify" if needs else END


async def _clarify_node(state) -> dict:
    msg = (
        (state.get("clarification_message") if isinstance(state, dict) else state.clarification_message)
        or "Could not find the requested data. Please refine your query."
    )
    return {"final_response": msg}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the v2 async LangGraph pipeline."""
    warm_up()

    graph = StateGraph(AgentState)

    graph.add_node("extractor", extractor)
    graph.add_node("retrieval", retrieval)
    graph.add_node("reason_and_synth", reason_and_synthesize)
    graph.add_node("clarify", _clarify_node)

    graph.set_entry_point("extractor")
    graph.add_edge("extractor", "retrieval")
    graph.add_edge("retrieval", "reason_and_synth")
    graph.add_conditional_edges(
        "reason_and_synth",
        _route_judge,
        {"clarify": "clarify", END: END},
    )
    graph.add_edge("clarify", END)

    return graph.compile()


def create_initial_state(user_query: str) -> AgentState:
    """Create the initial AgentState for a new query."""
    return AgentState(user_query=user_query)
