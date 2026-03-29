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

## Multi-Agent Workflow with LangGraph

```text
User query (Streamlit)
        │
        ▼
╔═══════════════════════════════════════════════════════════╗
║  Node 1 — extractor          [PARALLEL inside node]      ║
║                                                           ║
║   ┌─────────────────────┐   ┌───────────────────────────┐ ║
║   │  LLM call           │   │  Speculative Polars       │ ║
║   │  intent + filters   │   │  prefetch (7 aggregations)│ ║
║   │  → JSON schema      │   │  while LLM is running     │ ║
║   └──────────┬──────────┘   └─────────────┬─────────────┘ ║
║              └──────── asyncio.gather ─────┘              ║
╚═══════════════════════════╤═══════════════════════════════╝
                            │ intent + filters + prefetch_results
                            ▼  (sequential: node 2 waits for node 1)
╔═══════════════════════════════════════════════════════════╗
║  Node 2 — retrieval          [single path, ~0–10 ms]     ║
║                                                           ║
║   TTLCache hit?  →  <5 ms   (skip everything)            ║
║   Prefetch hit?  →  ~0 ms   (use prefetch result)        ║
║   Cache miss     →  ~10 ms  (fresh Polars query)         ║
║   + fuzzy name resolution (property/tenant/ledger)       ║
╚═══════════════════════════╤═══════════════════════════════╝
                            │ raw_results
                            ▼  (sequential: node 3 waits for node 2)
╔═══════════════════════════════════════════════════════════╗
║  Node 3 — reason_and_synthesize  [PARALLEL inside node]  ║
║                                                           ║
║   ┌─────────────────────┐   ┌───────────────────────────┐ ║
║   │  Critic + Judge     │   │  Synthesizer LLM          │ ║
║   │  Python, <1 ms      │   │  generates NL response    │ ║
║   │  validates filters  │   │  while critic runs        │ ║
║   └──────────┬──────────┘   └─────────────┬─────────────┘ ║
║         asyncio.create_task ───────────────┘              ║
║   If judge=CLARIFY → cancel synthesis, return hint        ║
║   If judge=OK      → return synthesizer response          ║
╚═══════════════════════════╤═══════════════════════════════╝
                            │
                            ▼
                  final_response → Streamlit UI
```

> **Sequential between nodes, parallel within nodes 1 and 3.** LangGraph executes nodes in directed order (1 → 2 → 3). The parallelism (`asyncio.gather` / `asyncio.create_task`) happens inside Node 1 (LLM + Polars prefetch run simultaneously) and Node 3 (critic + synthesizer run simultaneously).

**Graph edges:**

- `extractor` → `retrieval` (always)
- `retrieval` → `reason_and_synthesize` (always)
- `reason_and_synthesize` → `END` (always — clarification or answer)

**State** (`AgentState`): `user_query`, `intent`, `filters`, `prefetch_results`, `raw_results`, `critique`, `needs_clarification`, `final_response`, `cache_hit`, `iterations`, `timings`

---

## Challenges

### LLM Output vs Data Schema Mismatch

The LLM outputs human-readable values (e.g. `"Bank charges"`) but the parquet stores snake_case (`bank_charges`). Fixed by normalising filter values in `resolve_filters()` before querying, passing exact known values in the extractor system prompt, and adding a fuzzy fallback in `_apply_filters()`.

### Streamlit + asyncio Conflict

Streamlit runs its own event loop. `asyncio.run()` inside a Streamlit script raises `RuntimeError: This event loop is already running`. Fixed by scoping `asyncio.run()` only to the LangGraph `ainvoke` call and moving LLM pre-warm to a background daemon thread.

### Slow App Startup (60+ seconds)

`asyncio.run(prewarm_llm())` at module level blocked the entire UI while the GGUF model loaded. Fixed by moving pre-warm to `threading.Thread(daemon=True)` — UI is immediately responsive.

### Graph Recompilation on Every Rerun

`StateGraph` compilation with a Pydantic schema is expensive. `st.session_state` still re-ran it on each Streamlit rerun. Fixed with `@st.cache_resource` — compiled once per server process.

### Fuzzy Matching False Positives

Default `difflib` cutoff of 0.6 matched `"Building 999"` → `"Building 17"`. Raised cutoff to 0.80 for property and tenant resolution.

### Pydantic v2 Deprecation

`class Config` inside `BaseModel` is deprecated in Pydantic v2. Replaced with `model_config = ConfigDict(arbitrary_types_allowed=True)`.

---

## Data

`cortex.parquet` — 3,924 rows · 5 properties (Building 17/120/140/160/180) · 2024-Q1 to 2025-Q1 · columns: `entity_name`, `property_name`, `tenant_name`, `ledger_type`, `ledger_group`, `ledger_category`, `ledger_code`, `ledger_description`, `month`, `quarter`, `year`, `profit`.

See `data.md` for full schema and statistics, `SPEC.md` for the complete system specification.



[Strimlit UI](https://github.com/user-attachments/assets/a994adb0-5d94-4719-a1b5-0549e411b6ea)

