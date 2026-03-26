"""PropCo Financial Assistant — Streamlit UI (3-layer optimized architecture)."""

from __future__ import annotations

import asyncio

import streamlit as st

st.set_page_config(
    page_title="PropCo Financial Assistant",
    page_icon="🏢",
    layout="wide",
)

st.markdown("""
<style>
/* Circular send button */
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 50% !important;
    width: 2.6rem !important;
    height: 2.6rem !important;
    min-width: unset !important;
    padding: 0 !important;
    font-size: 1.1rem !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Imports (after page config)
# ---------------------------------------------------------------------------

import polars as pl
import plotly.express as px

from src.cache import cache_size
from src.data.loader import get_cache, warm_up
from src.graph.workflow import build_graph, create_initial_state
from src.utils.config import settings


@st.cache_resource
def _get_graph():
    """Compile the LangGraph pipeline exactly once for the lifetime of the server."""
    return build_graph()

# Warm up data caches on first load (fast — in-memory Parquet read)
warm_up()

# Pre-warm LLM in background — never blocks UI startup
if "llm_warmed" not in st.session_state:
    st.session_state.llm_warmed = True
    import threading as _threading
    import asyncio as _asyncio
    from src.llm.local import prewarm_llm as _prewarm_llm
    _threading.Thread(
        target=lambda: _asyncio.run(_prewarm_llm()),
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_chart(df_pd, intent: str) -> None:
    """Render an auto-selected Plotly bar chart based on intent."""
    try:
        if "total_profit" not in df_pd.columns:
            return
        if intent in ("pl_summary", "property_detail") and "property_name" in df_pd.columns:
            fig = px.bar(
                df_pd.sort_values("total_profit", ascending=True),
                x="total_profit", y="property_name", orientation="h",
                title="P&L by Property", labels={"total_profit": "Profit (€)", "property_name": "Property"},
                color="total_profit", color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig, width='stretch')
        elif intent == "comparison" and "property_name" in df_pd.columns and "ledger_type" in df_pd.columns:
            fig = px.bar(
                df_pd, x="property_name", y="total_profit", color="ledger_type",
                barmode="group", title="Revenue vs Expenses by Property",
                labels={"total_profit": "Profit (€)", "property_name": "Property"},
            )
            st.plotly_chart(fig, width='stretch')
        elif intent == "tenant_query" and "tenant_name" in df_pd.columns:
            top = df_pd.nlargest(10, "total_profit")
            fig = px.bar(
                top.sort_values("total_profit", ascending=True),
                x="total_profit", y="tenant_name", orientation="h",
                title="Top 10 Tenants by Profit", labels={"total_profit": "Profit (€)", "tenant_name": "Tenant"},
            )
            st.plotly_chart(fig, width='stretch')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


_init_session()


# ---------------------------------------------------------------------------
# Sidebar — KPIs + settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🏢 PropCo Dashboard")

    profit_by_year = get_cache("total_profit_by_year") or {}
    for year in sorted(profit_by_year, reverse=True):
        label = f"Total P&L {year}"
        st.metric(label, f"€{profit_by_year[year]:,.0f}")

    st.divider()
    props = sorted(get_cache("available_properties") or [])
    tenants = sorted(get_cache("available_tenants") or [])
    st.caption(f"🏗 Properties ({len(props)}): {', '.join(props)}")
    st.caption(f"👥 Tenants: {len(tenants)}")
    st.caption(f"📅 2024-Q1 — 2025-Q1")

    st.divider()
    st.subheader("⚙ Settings")
    backend = settings.llm_backend
    model = settings.local_model_path if backend == "local" else settings.openai_model_reasoning
    st.caption(f"LLM backend: `{backend}`")
    st.caption(f"Model: `{model}`")
    show_raw = st.toggle("Show retrieved data", value=True)

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_result = None
        # No st.rerun() — button click already triggers one rerun; a second is wasteful


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("🏢 PropCo Financial Assistant")

left_col, right_col = st.columns([3, 2])

# ---------------------------------------------------------------------------
# Chat panel (left)
# ---------------------------------------------------------------------------

with left_col:
    pending_prompt = st.session_state.pending_prompt

    # --- Quick picks: above the input ---
    st.markdown("**Quick picks:**")
    examples = [
        "Compare revenue for Building 17 vs 120 in 2024",
        "What's the total P&L for Q2 2024?",
        "Which tenant generates the most revenue?",
        "Building 140 expenses breakdown",
        "Bank charges in 2024",
    ]
    _qcols = st.columns(2)
    for i, ex in enumerate(examples):
        if _qcols[i % 2].button(ex, key=f"ex_{ex}"):
            st.session_state.pending_prompt = ex
            st.rerun()

    st.divider()

    # --- Input below quick picks ---
    with st.form(key="chat_form", clear_on_submit=True):
        _ic, _bc = st.columns([11, 1])
        with _ic:
            chat_input = st.text_input(
                "Ask about PropCo's financials...",
                placeholder="e.g. Bank charges in 2024, Compare Building 17 vs 120...",
                label_visibility="collapsed",
            )
        with _bc:
            submitted = st.form_submit_button("→")

    # Typed input takes priority; pending_prompt (example click) fires immediately
    prompt = chat_input.strip() if submitted and chat_input.strip() else None
    if not prompt and pending_prompt:
        prompt = pending_prompt
        st.session_state.pending_prompt = None

    st.divider()

    # --- Execute query (renders inline; on rerun appears in reversed history) ---
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                initial_state = create_initial_state(user_query=prompt)
                result = asyncio.run(
                    _get_graph().ainvoke(initial_state)
                )
                st.session_state.last_result = result

            _r = result if isinstance(result, dict) else result.model_dump()
            output = _r.get("final_response") or "No response generated."
            st.markdown(output)

            timings = _r.get("timings") or {}
            total_ms = sum(timings.values()) * 1000
            cache_hit = _r.get("cache_hit", False)
            intent = _r.get("intent") or "?"
            badge = (
                f"⚡ **{total_ms:.0f} ms** &nbsp;|&nbsp; "
                f"{'🟢 cache hit' if cache_hit else '🔵 live query'} &nbsp;|&nbsp; "
                f"intent: `{intent}`"
            )
            st.caption(badge)

        st.session_state.messages.append({"role": "assistant", "content": output})
        st.rerun()

    # --- Chat history: newest pair at top, oldest at bottom ---
    # Pair messages as [(user_q, asst_a), ...] then reverse so newest is first
    _msgs = st.session_state.messages
    _pairs = [(_msgs[i], _msgs[i + 1]) for i in range(0, len(_msgs) - 1, 2)]
    for _q, _a in reversed(_pairs):
        with st.chat_message(_q["role"]):
            st.markdown(_q["content"])
        with st.chat_message(_a["role"]):
            st.markdown(_a["content"])
    # Unpaired trailing user message (mid-flight, shouldn't normally appear)
    if len(_msgs) % 2 == 1:
        with st.chat_message(_msgs[-1]["role"]):
            st.markdown(_msgs[-1]["content"])


# ---------------------------------------------------------------------------
# Data panel (right)
# ---------------------------------------------------------------------------

with right_col:
    result = st.session_state.last_result
    _r = (result if isinstance(result, dict) else result.model_dump()) if result else {}

    if _r:
        intent = _r.get("intent")
        filters = _r.get("filters") or {}
        critique = _r.get("critique", "OK")
        needs_clarification = _r.get("needs_clarification", False)
        cache_hit = _r.get("cache_hit", False)
        timings = _r.get("timings") or {}

        st.subheader("📋 Query Details")
        if intent:
            st.caption(f"**Intent:** `{intent}`")
        if filters:
            st.caption(f"**Filters:** `{filters}`")
        if timings:
            timing_str = "  ·  ".join(f"{k}: {v*1000:.0f}ms" for k, v in timings.items())
            st.caption(f"⏱ {timing_str}")
        if cache_hit:
            st.success(f"🟢 Cache hit — {cache_size()} entries cached")
        if critique and critique != "OK":
            st.warning(f"⚠️ {critique}")
        if needs_clarification:
            st.info(_r.get("clarification_message", ""))

    if _r.get("raw_results") and show_raw:
        rows = _r["raw_results"]
        intent = _r.get("intent")
        st.subheader("� Retrieved Data")
        try:
            df_pl = pl.DataFrame(rows)
            df_pd = df_pl.to_pandas()
            st.dataframe(df_pd, width='stretch')

            # Auto Plotly chart for visual intents
            if intent in ("pl_summary", "comparison", "tenant_query", "property_detail"):
                _render_chart(df_pd, intent)
        except Exception:
            st.json(rows[:20])

    if _r.get("error"):
        st.error(f"❌ {_r['error']}")


# ---------------------------------------------------------------------------
# Pipeline stages (bottom)
# ---------------------------------------------------------------------------

st.divider()
_v2_stages = ["extractor", "retrieval", "reason_and_synth"]
_v2_labels = ["Extractor ⚡", "Retrieval 🗄", "Reason+Synth 🧠"]
_stage_cols = st.columns(len(_v2_stages))
_last = st.session_state.last_result
_lr = (_last if isinstance(_last, dict) else _last.model_dump()) if _last else {}
for _col, _stage, _label in zip(_stage_cols, _v2_stages, _v2_labels):
    _ms = _lr.get("timings", {}).get(_stage.replace("reason_and_synth", "reason_and_synthesize"), None)
    if _lr.get("intent") is not None:
        _badge = f" ({_ms*1000:.0f}ms)" if _ms is not None else ""
        _col.markdown(f"✅ **{_label}**{_badge}")
    else:
        _col.markdown(f"⬜ {_label}")


