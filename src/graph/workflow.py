"""LangGraph parallel multi-agent pipeline — v3.

Topology
--------
Layer 1  : planner  — LLM decomposes query + speculative Polars prefetch (parallel)
Fan-out 1: planner  → Send(worker) × N          — workers run in parallel via Send()
Layer 2  : aggregator — barrier; merges all worker_results → raw_results
Fan-out 2: aggregator → [critic || synthesizer]  — parallel validation + response
Barrier  : gate     — joins critic + synthesizer, applies caveat or clarification
Exit     : END  or  clarify → END

All nodes are async def. Timings merged via _merge_dicts reducer on AgentState.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.aggregator import aggregator
from src.agents.critic import critic
from src.agents.gate import gate
from src.agents.planner import planner
from src.agents.synthesizer import synthesizer
from src.agents.worker import worker
from src.data.loader import warm_up
from src.graph.state import AgentState


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_to_workers(state: dict) -> list[Send]:
    """Fan-out: one Send per worker assignment, fallback to single pl_worker."""
    assignments = state.get("worker_assignments") or []
    if not assignments:
        return [Send("worker", {
            **state,
            "current_assignment": {
                "worker_type": "pl_worker",
                "filters": state.get("filters") or {},
                "question": state.get("user_query", ""),
            },
        })]
    return [Send("worker", {**state, "current_assignment": a}) for a in assignments]


def _route_gate(state: dict) -> str:
    return "clarify" if state.get("needs_clarification") else END


async def _clarify_node(state: dict) -> dict:
    msg = (
        state.get("clarification_message")
        or "Could not find the requested data. Please refine your query."
    )
    return {"final_response": msg}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the v3 parallel multi-agent LangGraph pipeline."""
    warm_up()

    g = StateGraph(AgentState)

    g.add_node("planner", planner)
    g.add_node("worker", worker)
    g.add_node("aggregator", aggregator)
    g.add_node("critic", critic)
    g.add_node("synthesizer", synthesizer)
    g.add_node("gate", gate)
    g.add_node("clarify", _clarify_node)

    # Layer 1 → fan-out to parallel workers
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", _route_to_workers, ["worker"])

    # Workers converge at aggregator
    g.add_edge("worker", "aggregator")

    # Fan-out 2: parallel critic + synthesizer
    g.add_edge("aggregator", "critic")
    g.add_edge("aggregator", "synthesizer")
    g.add_edge("critic", "gate")
    g.add_edge("synthesizer", "gate")

    # ── Conditional exit ────────────────────────────────────────────────
    g.add_conditional_edges(
        "gate",
        _route_gate,
        {"clarify": "clarify", END: END},
    )
    g.add_edge("clarify", END)

    return g.compile()


def create_initial_state(
    user_query: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict:
    """Create the initial state dict for a new query."""
    return {
        "user_query": user_query,
        "conversation_history": conversation_history or [],
        "worker_results": [],
        "timings": {},
    }
