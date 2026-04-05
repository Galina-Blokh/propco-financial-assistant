# PropCo Financial Assistant

A **parallel multi-agent** real estate asset management assistant built with **LangGraph**. Six async agents orchestrated via fan-out edges deliver answers about P&L, tenants, buildings, and ledger entries in ~350–1,000 ms. Features **multi-turn conversation memory**, **true OpenAI token streaming**, and **ambiguity detection with clarification**.

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env — set OPENAI_API_KEY

# 2. Install dependencies
uv sync

# 3. Run the app (Streamlit only)
uv run streamlit run ui/streamlit_app.py --server.headless=true --server.port=8501

# Or: Streamlit + MLflow UI together (recommended for telemetry)
bash scripts/run_demo.sh
```

Open **http://localhost:8501** (chat). With `run_demo.sh`, MLflow is **http://localhost:5000**.

The UI loads **without** importing LangGraph first, so the page should appear in **seconds**. The **first** chat message triggers a one-time graph compile (can take **~30–60s** on WSL or slow disks); later queries reuse the cached graph.

## Architecture (v3 — parallel fan-out)

```text
                   ┌────────────────────────────────────────────────────┐
                   │          LangGraph AsyncStateGraph                  │
                   │                                                    │
  user_query → START                                                    │
                   │                                                    │
          ┌────────┴────────┐     ← Fan-out 1: parallel                │
          ▼                 ▼                                           │
     [extractor]      [prefetcher]                                     │
     (gpt-4o-mini:     (6 Polars                                       │
      intent+filters    aggregations)                                  │
      +ambiguity)                                                      │
          │                 │                                           │
          └────────┬────────┘                                           │
                   ▼                                                    │
            [join_layer1]                                               │
              │         │                                               │
              ▼         ▼   ← Conditional: ambiguity → [clarify]→END   │
         [retriever]                                                    │
         L1 cache / L2 prefetch / L3 Polars                            │
              │                                                         │
     ┌────────┴────────┐     ← Fan-out 2: parallel                     │
     ▼                 ▼                                                │
  [critic]      [synthesizer]                                          │
  (<1 ms)        (true OpenAI stream)                                  │
     │                 │                                                │
     └────────┬────────┘                                                │
              ▼                                                         │
           [gate]  → END  or  → [clarify] → END                        │
              └────────────────────────────────────────────────────────┘
```

**2 LLM calls** (extraction: gpt-4o-mini, synthesis: gpt-4o with streaming) · Cache hit: **<5 ms** · Prefetch covers ~85% of queries at 0 ms retrieval cost · **Multi-turn memory**: last 3 turns passed to extractor + synthesizer.

## Performance Targets

| Metric | Target |
|---|---|
| First data table | ~200 ms |
| First prose token | ~350 ms |
| Cache hit (full) | <5 ms |
| Uncached query | ~550–1,000 ms |

## LLM Backends

| Backend | Config | Notes |
|---|---|---|
| OpenAI (default) | `LLM_BACKEND=openai` + `OPENAI_API_KEY` | Extraction: gpt-4o-mini; Synthesis: gpt-4o with stream=True |
| Local GGUF | `LLM_BACKEND=local` + `LOCAL_MODEL_PATH` | llama-cpp-python, no API cost |

## Project Structure

```text
src/
  agents/
    extractor.py        Layer 1a — LLM intent + filter parsing + ambiguity detection
    prefetcher.py       Layer 1b — 6 parallel Polars speculative prefetch slabs
    join_layer1.py      Layer 1 join barrier
    retriever.py        Layer 2  — 3-tier cache → Polars query router
    critic.py           Layer 3a — deterministic data validator (<1 ms)
    synthesizer.py      Layer 3b — LLM prose + true OpenAI token streaming
    gate.py             Layer 3  — join + conditional routing
  graph/
    state.py            AgentState (TypedDict + merge reducer + conversation_history)
    workflow.py         build_graph() — fan-out topology with START/END
  data/
    loader.py           Singleton DataFrame + startup caches
    fuzzy.py            Fuzzy name resolution (difflib)
  llm/
    local.py            Dual-mode LLM: chat_complete + chat_complete_stream; per-call model routing
  cache.py              TTLCache singleton
  utils/
    config.py           Pydantic v2 BaseSettings from .env
    llm_client.py       get_llm_client() — event-loop-aware AsyncOpenAI (Streamlit-safe)
ui/
  streamlit_app.py      Two-phase streaming UI
tests/                  Unit + integration tests
```

## Configuration

All settings via `.env` (see `.env.example`):

```bash
OPENAI_API_KEY=sk-...
LLM_BACKEND=openai              # or "local"
OPENAI_MODEL_REASONING=gpt-4o   # synthesis (streamed)
OPENAI_MODEL_EXTRACTION=gpt-4o-mini  # extraction (fast JSON)
CORTEX_DATA_PATH=./cortex.parquet
CACHE_TTL_SECONDS=3600
# Local GGUF only:
LOCAL_MODEL_PATH=/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

## MLflow Performance Dashboard

Each chat answer is logged as a **classic MLflow experiment run** (parameters + metrics, and optional text artifacts). This is **not** the same as the Model Registry, Tracing, Prompt Registry, or Evaluation workflows in the MLflow UI.

### Where your data actually shows up (and what stays empty)

| MLflow UI area | Why it is empty for this project |
|----------------|-----------------------------------|
| **Model Registry** | We never call `mlflow.register_model`. Registry is for versioned model packages, not chat logs. |
| **Traces** / GenAI tracing | We do not use MLflow’s tracing SDK or OpenAI autolog tracing. Only `start_run` + params/metrics. |
| **Prompts** (prompt registry) | We do not log prompts through MLflow’s prompt store API. |
| **Evaluations** | We do not run MLflow’s evaluation jobs. |
| **Experiments → `real_estate_agent` → Runs** | **This is where your queries appear.** Open a run to see **Parameters**, **Metrics**, and (if enabled) **Artifacts**. |

### Start the UI with the same store as the app

Use the URI copied from the Streamlit sidebar (**MLflow (tracking URI)**). Example (default file store on WSL):

```bash
cd /path/to/graph_rag
uv run mlflow ui --backend-store-uri 'file:///mnt/c/Users/you/CascadeProjects/graph_rag/mlflow-tracking' --host 0.0.0.0 --port 5000
```

`scripts/run_demo.sh` exports that `file://…` URI for you, starts MLflow on **5000**, then Streamlit on **8501**. Override ports: `MLFLOW_PORT=5001 STREAMLIT_PORT=8502 bash scripts/run_demo.sh`.

After sending a chat message, go to **Experiments** → **`real_estate_agent`** → click a **run** → **Parameters** / **Metrics**. Prompt/response **files** under **Artifacts** only appear if you set **`MLFLOW_LOG_ARTIFACTS=true`** in `.env`.

### Configuration (`.env`)

```bash
# Omit MLFLOW_TRACKING_URI to use <repo>/mlflow-tracking/ (recommended)
MLFLOW_EXPERIMENT_NAME=real_estate_agent
MLFLOW_ENABLED=true
MLFLOW_LOG_ARTIFACTS=false
```

**If Runs are still missing:** app and `mlflow ui` must share the **exact** `--backend-store-uri`. Remove a conflicting `MLFLOW_TRACKING_URI=sqlite:///...` from `.env` if you want the file default, or point both sides at the same resolved path. Set **`MLFLOW_ENABLED=false`** if logging interferes with the UI.

Logging uses a **single background worker** queue so SQLite/file IO does not block Streamlit.

## Testing

```bash
# Unit tests (no API key needed)
uv run pytest tests/ --ignore=tests/test_e2e.py -v

# End-to-end (requires OPENAI_API_KEY in .env)
uv run pytest tests/test_e2e.py -v
```

## Data

`cortex.parquet` — 3,924 rows · 5 properties (Building 17/120/140/160/180) · 2024-Q1 to 2025-Q1 · columns: `entity_name`, `property_name`, `tenant_name`, `ledger_type`, `ledger_group`, `ledger_category`, `ledger_code`, `ledger_description`, `month`, `quarter`, `year`, `profit`.

Documentation: `SPEC.md` (system spec), `data.md` (Cortex dataset schema), `docs/decisions.md` (ADRs), `docs/implementation_plan.md` (milestones, deployment checklist, backlog).
