# PropCo Financial Assistant

A **3-node async LangGraph pipeline** for natural-language queries over real estate financial data. Ask questions about P&L, tenants, buildings, and ledger entries — answers in ~350–1,100 ms.

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env — set OPENAI_API_KEY

# 2. Install dependencies
uv sync

# 3. Run the app
PYTHONPATH=. uv run streamlit run ui/streamlit_app.py --server.headless=true
```

Open **http://localhost:8502**

## Architecture (v2 — async-first)

```text
┌─────────────────────────────────────────────────────────────────┐
│  Node 1: extractor                                              │
│    LLM intent+filters extraction  ║  Polars speculative prefetch│
│    (asyncio.gather — parallel)                                  │
├─────────────────────────────────────────────────────────────────┤
│  Node 2: retrieval                                              │
│    Tier 1: TTLCache hit   (<5 ms)                               │
│    Tier 2: Prefetch hit   (~0 ms)                               │
│    Tier 3: Fresh Polars   (~10 ms)                              │
├─────────────────────────────────────────────────────────────────┤
│  Node 3: reason_and_synthesize                                  │
│    Critic+Judge (<1ms Python)  ║  Synthesizer LLM              │
│    (asyncio.create_task — parallel)                             │
└─────────────────────────────────────────────────────────────────┘
```

**2 LLM calls total** · Cache hit serves full response in **<5 ms** · Speculative prefetch covers ~85 % of queries at 0 ms retrieval cost.

## Performance Targets

| Metric | Target |
|---|---|
| First data table | ~200 ms |
| First prose token | ~350 ms |
| Cache hit (full) | <5 ms |
| Uncached query | ~400–1,100 ms |

## LLM Backends

| Backend | Config | Notes |
|---|---|---|
| OpenAI (default) | `LLM_BACKEND=openai` + `OPENAI_API_KEY` | gpt-4o-mini |
| Local GGUF | `LLM_BACKEND=local` + `LOCAL_MODEL_PATH` | llama-cpp-python, no API cost |

## Project Structure

```text
src/
  cache.py              TTLCache(maxsize=256, ttl=300s) — response cache
  agents/
    extractor.py        Node 1: LLM JSON extraction + speculative prefetch
    retrieval.py        Node 2: 3-tier cache → Polars query router
    reason_and_synthesize.py  Node 3: parallel critic+judge + LLM synthesis
  graph/
    state.py            AgentState (Pydantic BaseModel)
    workflow.py         LangGraph: extractor → retrieval → reason_and_synth
  data/
    loader.py           Singleton DataFrame + 8 startup caches (warm_up)
    fuzzy.py            Fuzzy name resolution (property, tenant, ledger)
  llm/
    local.py            Unified LLM interface: OpenAI + GGUF (asyncio.Lock)
  utils/
    config.py           Pydantic settings from .env
ui/
  streamlit_app.py      Chat UI — quick picks, reverse-chronological history
scripts/
  preprocess_data.py    Parquet schema inspection + optimized file writer
tests/                  56 tests, 0 failures
```

## Configuration

All settings via `.env` (see `.env.example`):

```bash
OPENAI_API_KEY=sk-...
LLM_BACKEND=openai             # or "local"
OPENAI_MODEL_REASONING=gpt-4o-mini
CORTEX_DATA_PATH=./cortex.parquet
# Local GGUF only:
LOCAL_MODEL_PATH=/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
LOCAL_N_GPU_LAYERS=35
```

## Testing

```bash
PYTHONPATH=. uv run pytest tests/ --ignore=tests/test_e2e.py -v
```

## Data

`cortex.parquet` — 3,924 rows · 5 properties (Building 17/120/140/160/180) · 2024-Q1 to 2025-Q1 · columns: `entity_name`, `property_name`, `tenant_name`, `ledger_type`, `ledger_group`, `ledger_category`, `ledger_code`, `ledger_description`, `month`, `quarter`, `year`, `profit`.

See `data.md` for full schema and statistics, `SPEC.md` for the complete system specification.
