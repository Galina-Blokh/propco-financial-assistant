"""PropCo Financial Assistant — Streamlit UI with two-phase rendering.

Phase 1: Data table rendered from raw_results after graph completes.
Phase 2: Prose streamed token-by-token from the synthesizer's queue.
Telemetry badge shown under each Q&A pair.
"""

from __future__ import annotations

import asyncio

import streamlit as st

st.set_page_config(
    page_title="PropCo Financial Assistant",
    page_icon="🏢",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Light imports only — LangGraph + all agents are imported lazily on first query
# (eager import can block 30–120s on WSL / slow disks and leaves a white page).
# ---------------------------------------------------------------------------

import polars as pl

from src.data.loader import get_cache, warm_up
from src.utils.config import settings


# ---------------------------------------------------------------------------
# One-time initialisation
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_graph():
    """Compile the LangGraph pipeline exactly once (heavy import happens here)."""
    from src.telemetry import _ensure_init
    _ensure_init()  # Enable mlflow.langchain.autolog() before first graph invocation
    from src.graph.workflow import build_graph
    return build_graph()


try:
    warm_up()
except Exception as exc:
    st.error(
        f"**Could not load financial data.** Check `CORTEX_DATA_PATH` in `.env` "
        f"and that `cortex.parquet` exists.\n\n`{type(exc).__name__}: {exc}`"
    )
    st.stop()

if "llm_warmed" not in st.session_state:
    st.session_state.llm_warmed = True
    import threading as _threading
    from src.llm.local import prewarm_llm as _prewarm_llm

    _threading.Thread(
        target=lambda: asyncio.run(_prewarm_llm()), daemon=True
    ).start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_chart(container, df_pd, intent: str | None, key_suffix: str = "") -> None:
    """Render an auto-selected Plotly bar chart based on intent."""
    try:
        import plotly.express as px

        if "total_profit" not in df_pd.columns:
            return
        if intent in ("pl_summary", "property_detail") and "property_name" in df_pd.columns:
            fig = px.bar(
                df_pd.sort_values("total_profit", ascending=True),
                x="total_profit",
                y="property_name",
                orientation="h",
                title="P&L by Property",
                labels={"total_profit": "Profit (€)", "property_name": "Property"},
                color="total_profit",
                color_continuous_scale="RdYlGn",
            )
            container.plotly_chart(fig, key=f"chart_pl{key_suffix}", width="stretch")
        elif (
            intent == "comparison"
            and "property_name" in df_pd.columns
            and "ledger_type" in df_pd.columns
        ):
            fig = px.bar(
                df_pd,
                x="property_name",
                y="total_profit",
                color="ledger_type",
                barmode="group",
                title="Revenue vs Expenses by Property",
                labels={"total_profit": "Profit (€)", "property_name": "Property"},
            )
            container.plotly_chart(fig, key=f"chart_cmp{key_suffix}", width="stretch")
        elif intent == "tenant_query" and "tenant_name" in df_pd.columns:
            top = df_pd.nlargest(10, "total_profit")
            fig = px.bar(
                top.sort_values("total_profit", ascending=True),
                x="total_profit",
                y="tenant_name",
                orientation="h",
                title="Top 10 Tenants by Profit",
                labels={"total_profit": "Profit (€)", "tenant_name": "Tenant"},
            )
            container.plotly_chart(fig, key=f"chart_tenant{key_suffix}", width="stretch")
    except Exception:
        pass


def _emit_mlflow_log(result: dict, user_query: str) -> None:
    """Call ``enqueue_query_log`` if present, else ``log_query`` (older trees)."""
    try:
        import src.telemetry as _tel
    except ImportError:
        return
    fn = getattr(_tel, "enqueue_query_log", None) or getattr(_tel, "log_query", None)
    if callable(fn):
        fn(result, user_query)


def _render_telemetry_badge(timings: dict, cache_tier: str | None, intent: str | None, workers: list | None = None) -> None:
    """Render the per-query telemetry line."""
    total_ms = sum(timings.values()) * 1000
    tier = cache_tier or "—"
    intent_str = intent or "?"
    worker_keys = [k for k in timings if k.startswith("worker_")]
    workers_str = " + ".join(k[7:] for k in worker_keys) if worker_keys else (" + ".join(workers) if workers else "")
    non_worker_timings = {k: v for k, v in timings.items() if not k.startswith("worker_")}
    detail = " · ".join(f"{k}: {v * 1000:.0f}ms" for k, v in non_worker_timings.items())
    workers_part = f" | workers: `{workers_str}`" if workers_str else ""
    st.caption(
        f"⚡ **{total_ms:.0f} ms** | cache: {tier} | "
        f"intent: `{intent_str}`{workers_part} | {detail}"
    )


def _render_history_section(msgs: list) -> None:
    """Render completed Q&A pairs, newest first (reverse chronological by pair)."""
    pair_count = len(msgs) // 2
    for pair_idx in range(pair_count - 1, -1, -1):
        idx = pair_idx * 2
        q = msgs[idx]
        a = msgs[idx + 1] if idx + 1 < len(msgs) else None

        with st.chat_message("user"):
            st.markdown(q["content"])

        if a:
            with st.chat_message("assistant"):
                st.markdown(a["content"])

                meta = a.get("meta") or {}
                raw = meta.get("raw_results")
                if raw:
                    try:
                        df_pd = pl.DataFrame(raw).to_pandas()
                        with st.expander("📊 Retrieved Data", expanded=False):
                            st.dataframe(df_pd, width="stretch")
                        _render_chart(st, df_pd, meta.get("intent"), key_suffix=f"_h{idx}")
                    except Exception:
                        pass

                _render_telemetry_badge(
                    meta.get("timings") or {},
                    meta.get("cache_tier"),
                    meta.get("intent"),
                    meta.get("workers"),
                )

    if len(msgs) % 2 == 1 and msgs:
        with st.chat_message("user"):
            st.markdown(msgs[-1]["content"])




# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ---------------------------------------------------------------------------
# Sidebar — KPIs + settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🏢 PropCo Dashboard")

    profit_by_year = get_cache("total_profit_by_year") or {}
    for year in sorted(profit_by_year, reverse=True):
        st.metric(f"Total P&L {year}", f"€{profit_by_year[year]:,.0f}")

    st.divider()
    props = sorted(get_cache("available_properties") or [])
    tenants = sorted(get_cache("available_tenants") or [])
    st.caption(f"🏗 Properties ({len(props)}): {', '.join(props)}")
    st.caption(f"👥 Tenants: {len(tenants)}")
    st.caption("📅 2024-Q1 — 2025-Q1")

    st.divider()
    st.subheader("⚙ Settings")
    backend = settings.llm_backend
    model = (
        settings.local_model_path if backend == "local" else settings.openai_model_reasoning
    )
    st.caption(f"LLM: `{backend}` · Model: `{model}`")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []

    with st.expander("MLflow (tracking URI)", expanded=False):
        from src.telemetry import get_tracking_uri

        st.caption(
            "Use **Experiments** → experiment below → **Runs** (not Registry, Traces, or Evaluations)."
        )
        st.code(get_tracking_uri(), language=None)
        st.caption(
            f"Experiment: `{settings.mlflow_experiment_name}` · "
            f"enabled `{settings.mlflow_enabled}` · artifacts `{settings.mlflow_log_artifacts}`"
        )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("🏢 PropCo Financial Assistant")

# --- Input at the TOP ---
pending = st.session_state.pending_prompt
prompt = st.chat_input("Ask about PropCo's financials…")
if not prompt and pending:
    prompt = pending
    st.session_state.pending_prompt = None

# --- Quick-pick example buttons ---
st.markdown("**Try these:**")
examples = [
    "Compare revenue for Building 17 vs 120 in 2024",
    "What's the total P&L for Q2 2024?",
    "Which tenant generates the most revenue?",
    "Building 140 expenses breakdown",
    "Bank charges in 2024",
]


def _queue_example(text: str) -> None:
    st.session_state.pending_prompt = text


qcols = st.columns(len(examples))
for i, ex in enumerate(examples):
    qcols[i].button(ex, key=f"ex_{i}", on_click=_queue_example, args=(ex,))

st.divider()

# ---------------------------------------------------------------------------
# Active query — renders right below input (newest first)
# ---------------------------------------------------------------------------

if prompt:
    from src.graph.workflow import create_initial_state

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_ph = st.empty()
        prose_ph = st.empty()
        data_container = st.container()

        status_ph.caption("⏳ Processing…")

        async def _run():
            graph = _get_graph()
            token_queue: asyncio.Queue = asyncio.Queue()
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-6:]
            ]
            initial = create_initial_state(
                user_query=prompt, conversation_history=history
            )
            initial["token_queue"] = token_queue

            graph_task = asyncio.create_task(graph.ainvoke(initial))

            streamed = ""
            while True:
                try:
                    token = await asyncio.wait_for(token_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    if graph_task.done():
                        break
                    continue
                if token is None:
                    break
                streamed += token
                prose_ph.markdown(streamed + "▌")

            result = await graph_task
            return result, streamed

        async def _run_with_timeout():
            return await asyncio.wait_for(_run(), timeout=180.0)

        try:
            # Run async graph in a dedicated thread with its own event loop to
            # avoid RuntimeError when Streamlit already has a running event loop.
            import concurrent.futures as _cf
            import threading as _thr

            _result_box: list = [None]
            _exc_box: list = [None]

            def _run_in_thread():
                _loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_loop)
                try:
                    _result_box[0] = _loop.run_until_complete(_run_with_timeout())
                except Exception as _e:
                    _exc_box[0] = _e
                finally:
                    _loop.close()
                    asyncio.set_event_loop(None)

            _t = _thr.Thread(target=_run_in_thread, daemon=True)
            _t.start()
            _t.join(timeout=185)

            if _exc_box[0] is not None:
                raise _exc_box[0]
            if _result_box[0] is None:
                raise asyncio.TimeoutError

            result, streamed = _result_box[0]
            _r = result if isinstance(result, dict) else {}
            output = _r.get("final_response") or streamed or "No response generated."
        except asyncio.TimeoutError:
            _r = {}
            output = "Request timed out after 3 minutes — try a simpler question or check your network."
        except Exception as exc:
            _r = {}
            output = f"Something went wrong — please try again. ({type(exc).__name__})"

        prose_ph.markdown(output)
        status_ph.empty()

        if _r.get("raw_results"):
            try:
                df = pl.DataFrame(_r["raw_results"])
                df_pd = df.to_pandas()
                with data_container.expander("📊 Retrieved Data", expanded=False):
                    st.dataframe(df_pd, width="stretch")
                _render_chart(data_container, df_pd, _r.get("intent"), key_suffix="_live")
            except Exception:
                pass

        _render_telemetry_badge(
            _r.get("timings") or {},
            _r.get("cache_tier"),
            _r.get("intent"),
        )

    # Prior turns (current reply is above). Must run before st.rerun() — the block
    # below is skipped when rerun aborts the script, which hid all older Q&A.
    _render_history_section(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}})
    # Limit stored raw_results to 20 rows — prevents session state bloat after
    # many queries and is the primary cause of UI slowdown/stacking.
    _raw_trimmed = (_r.get("raw_results") or [])[:20]
    _worker_types = list(dict.fromkeys(
        wr.get("worker_type", "") for wr in (_r.get("worker_results") or [])
        if wr.get("worker_type")
    ))
    st.session_state.messages.append({
        "role": "assistant",
        "content": output,
        "meta": {
            "timings": _r.get("timings") or {},
            "cache_tier": _r.get("cache_tier"),
            "intent": _r.get("intent"),
            "raw_results": _raw_trimmed,
            "workers": _worker_types,
        },
    })
    _emit_mlflow_log(_r, prompt)
    st.rerun()

else:
    # Idle: show full transcript (newest pair first).
    _render_history_section(st.session_state.messages)
