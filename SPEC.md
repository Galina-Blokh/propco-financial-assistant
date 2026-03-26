# PropCo Multi-Agent Real Estate Assistant — Optimized Spec v2

## 1. Overview

An **ultra-low-latency, async-first, local-first** multi-agent system for real estate asset
management (PropCo). The system answers natural-language queries about P&L, tenants,
properties, and ledger data — orchestrated by LangGraph with fully async nodes, powered
by a local GGUF LLM, and served via a Streamlit UI.

**Core performance philosophy:**

- Every node in the LangGraph graph is `async`
- Independent work is always `asyncio.gather`'d — never sequentially awaited
- The LLM is warm before the first user query (background pre-warm at startup)
- Retrieval results render in the UI immediately, while synthesis streams in parallel
- Response caching eliminates LLM calls entirely for repeated queries

---

## 2. Architecture — Async Pipeline

```text
User (Streamlit UI)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LangGraph AsyncGraph                           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  PARALLEL FAST-PATH (asyncio.gather)                     │       │
│  │                                                          │       │
│  │  ┌─────────────────┐    ┌──────────────────────────────┐ │       │
│  │  │  Extractor      │    │  Speculative Pre-fetcher     │ │       │
│  │  │  (Local LLM     │    │  (Polars: run ALL common     │ │       │
│  │  │   async stream) │    │   aggregations in parallel   │ │       │
│  │  └────────┬────────┘    │   while LLM extracts intent) │ │       │
│  │           │             └──────────────┬───────────────┘ │       │
│  └───────────┼──────────────────────────┼──────────────────┘       │
│              │  intent+filters ready     │  prefetch cache ready    │
│              ▼                           ▼                          │
│         ┌──────────────────────────────────┐                        │
│         │  Smart Retrieval                 │                        │
│         │  (cache hit → 0 ms              │                        │
│         │   prefetch hit → 0 ms           │                        │
│         │   cache miss → Polars 5–50 ms)  │                        │
│         └──────────────┬───────────────────┘                        │
│                        │ raw_results → UI renders table NOW         │
│         ┌──────────────▼───────────────────────────────────┐        │
│         │  PARALLEL REASONING (asyncio.gather)             │        │
│         │                                                  │        │
│         │  ┌───────────────┐   ┌──────────────────────┐   │        │
│         │  │ Critic+Judge  │   │ Synthesizer (stream) │   │        │
│         │  │ (Python, <1ms)│   │ (Local LLM, tokens   │   │        │
│         │  └───────┬───────┘   │  pushed to UI queue) │   │        │
│         │          │           └──────────┬───────────┘   │        │
│         └──────────┼──────────────────────┼───────────────┘        │
│                    │                      │                         │
│         ┌──────────▼──────────────────────▼───────────┐            │
│         │  Gate: if judge=False → cancel synthesis    │            │
│         │         stream, emit clarification instead  │            │
│         └──────────────────────┬──────────────────────┘            │
└────────────────────────────────┼────────────────────────────────────┘
                                 ▼
                    Streamlit: table renders instantly
                    from raw_results; prose streams in
                    token-by-token via asyncio.Queue
```

### Key Async Wins vs v1

| What | v1 (Sequential) | v2 (Async/Parallel) | Saving |
| --- | --- | --- | --- |
| Extractor + Speculative Prefetch | Serial (prefetch never happened) | `gather` → run simultaneously | ~40–200 ms |
| Critic + Synthesizer start | Critic first, then Synth | `gather` → start Synth immediately, cancel if Critic fails | ~200–600 ms |
| Comparison sub-queries | One at a time | All property queries `gather`'d | ~20–100 ms |
| Repeated queries | Full pipeline every time | Cache hit → 0 ms LLM | ~350–1000 ms |
| UI data table | Waits for full response | Renders from `raw_results` before synthesis finishes | ~200–600 ms perceived |
| LLM warm-up | First query pays cold start | Background pre-warm task at startup | ~500–2000 ms first query |

---

## 3. Technology Stack

| Component | Technology | Rationale |
| --- | --- | --- |
| Orchestration | **LangGraph** `AsyncStateGraph` | Async-native graph; all nodes `async def` |
| Local LLM | **Llama-3.1-8B-Instruct Q4_K_M GGUF** via `llama-cpp-python` | Zero remote latency; async via `asyncio.to_thread` |
| Data Layer | **Polars** + **pre-loaded Parquet** | Columnar, SIMD-accelerated; 10–100x faster than pandas |
| Async runtime | **asyncio** + `asyncio.gather` / `asyncio.Queue` | Native Python async; no extra dependencies |
| LLM thread isolation | `asyncio.to_thread` wrapping llama-cpp calls | Prevents LLM inference blocking the event loop |
| Response cache | **cachetools `TTLCache`** keyed on `(intent, frozenset(filters))` | O(1) cache hit, skips entire LLM pipeline |
| UI | **Streamlit** + `asyncio.Queue` for token streaming | Tokens pushed to UI as they arrive |
| Containerization | **Docker** — GGUF model baked into image | Zero cold-start model download |
| Config | **Pydantic v2 Settings** | Type-safe env/config |
| Testing | **pytest-asyncio** | Full async test coverage |

---

## 4. Data Layer — Async-Safe, Pre-loaded

### 4.1 Startup Sequence (non-blocking)

All heavy I/O happens at application startup in a background `asyncio` task, before any
user query arrives:

```python
import asyncio
import polars as pl
from cachetools import TTLCache

# Global singletons — written once at startup, read-only thereafter
_DF: pl.DataFrame | None = None
_AGG_CACHE: dict = {}
_CACHE: TTLCache = TTLCache(maxsize=256, ttl=300)  # 5-min TTL
_STARTUP_DONE: asyncio.Event = asyncio.Event()

async def startup():
    """Run at app launch. Pre-loads data and warms the LLM concurrently."""
    await asyncio.gather(
        _load_dataframe(),
        _prewarm_llm(),          # LLM loads model weights into memory/VRAM
    )
    await _precompute_aggregations()
    _STARTUP_DONE.set()

async def _load_dataframe():
    global _DF
    _DF = await asyncio.to_thread(pl.read_parquet, "data/propco.parquet")

async def _precompute_aggregations():
    df = _DF
    results = await asyncio.gather(
        asyncio.to_thread(lambda: df.group_by("year").agg(pl.col("profit").sum()).to_dicts()),
        asyncio.to_thread(lambda: df.group_by("property_name").agg(pl.col("profit").sum()).to_dicts()),
        asyncio.to_thread(lambda: df.group_by("tenant_name").agg(pl.col("profit").sum()).to_dicts()),
        asyncio.to_thread(lambda: df.group_by("quarter").agg(pl.col("profit").sum()).to_dicts()),
        asyncio.to_thread(lambda: set(df["property_name"].drop_nulls().to_list())),
        asyncio.to_thread(lambda: set(df["tenant_name"].drop_nulls().to_list())),
    )
    global _AGG_CACHE
    _AGG_CACHE = {
        "by_year":     results[0],
        "by_property": results[1],
        "by_tenant":   results[2],
        "by_quarter":  results[3],
        "properties":  results[4],
        "tenants":     results[5],
    }

async def _prewarm_llm():
    """Run a no-op inference to load weights into memory before first real query."""
    llm = get_llm()
    await asyncio.to_thread(llm, "Hi", max_tokens=1)
```

### 4.2 Parquet Schema

12 columns: `entity_name`, `property_name`, `tenant_name`, `ledger_type`,
`ledger_group`, `ledger_category`, `ledger_code`, `ledger_description`,
`month`, `quarter`, `year`, `profit`

Pre-convert CSV → Parquet at Docker build time (not at runtime).

---

## 5. LangGraph State

```python
from typing import Any
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    # Input
    user_query: str

    # Layer 1 output
    intent: str | None = None
    # Intents: pl_summary | property_detail | tenant_query | comparison | ledger_query | general
    filters: dict[str, Any] = {}
    # Filter keys: property_name, tenant_name, year, quarter, month, ledger_group,
    #              ledger_category, ledger_type, property_names (list, for comparison)

    # Speculative prefetch output (populated in parallel with extraction)
    prefetch_results: dict[str, list[dict]] = {}

    # Layer 2 output
    raw_results: list[dict] | None = None
    retrieval_error: str | None = None
    cache_hit: bool = False

    # Layer 3 output
    critique: str | None = None
    judgment: bool = True
    final_response: str | None = None

    # Streaming — asyncio.Queue passed by reference; excluded from serialization
    token_queue: Any = Field(default=None, exclude=True)

    # Control flow
    needs_clarification: bool = False
    clarification_message: str | None = None
    iterations: int = 0

    # Telemetry
    timings: dict[str, float] = {}

    class Config:
        arbitrary_types_allowed = True
```

---

## 6. Layer 1 — Async Extractor + Speculative Prefetcher (Parallel)

This is the core async win: **while the LLM parses intent, Polars pre-fetches all
common aggregations in parallel**. By the time intent is known, the data is often
already ready — making retrieval effectively 0 ms.

```python
async def extractor(state: AgentState) -> dict:
    """Run LLM extraction and Polars speculative prefetch simultaneously."""
    t0 = time.perf_counter()

    (intent, filters), prefetch = await asyncio.gather(
        _extract_intent(state.user_query),
        _speculative_prefetch(),
    )

    return {
        "intent": intent,
        "filters": filters,
        "prefetch_results": prefetch,
        "timings": {"extractor": time.perf_counter() - t0},
    }
```

**Speculative prefetch keys (7 aggregations run simultaneously):**

| Key | Query |
| --- | --- |
| `pl_by_property_2024` | P&L grouped by property, year=2024 |
| `pl_by_property_2025` | P&L grouped by property, year=2025 |
| `by_tenant` | Profit grouped by tenant+property |
| `by_quarter` | Profit grouped by property+quarter |
| `comparison_2024` | Revenue+expenses by property+ledger_type, 2024 |
| `comparison_2025` | Revenue+expenses by property+ledger_type, 2025 |
| `pl_all` | P&L grouped by property, all years |

**Net latency for Layer 1:** `max(LLM extraction, Polars prefetch)` ≈ **150–400 ms**
instead of `LLM + Polars` sequentially ≈ 200–450 ms.

---

## 7. Layer 2 — Smart Async Retrieval

### 7.1 Three-Tier Cache: Response Cache → Prefetch → Fresh Query

```python
async def retrieval(state: AgentState) -> dict:
    cache_key = make_cache_key(state.intent, state.filters)

    # Tier 1: Response cache hit → 0 ms, skip all computation
    if (cached := cache_get(cache_key)) is not None:
        return {"raw_results": cached, "cache_hit": True, "timings": {"retrieval": 0.0}}

    # Tier 2: Prefetch hit → 0 ms (data computed during extraction)
    pkey = _prefetch_lookup_key(state.intent, state.filters)
    if pkey and (results := state.prefetch_results.get(pkey)):
        cache_set(cache_key, results)
        return {"raw_results": results, "cache_hit": False}

    # Tier 3: Fresh Polars query — run off event loop
    results = await asyncio.to_thread(_run_polars_query, state.intent, state.filters)
    cache_set(cache_key, results)
    return {"raw_results": results, "cache_hit": False}
```

### 7.2 Prefetch Key Mapping

| Intent + Year | Prefetch key consumed |
| --- | --- |
| `pl_summary` + 2024 | `pl_by_property_2024` |
| `pl_summary` + 2025 | `pl_by_property_2025` |
| `pl_summary` (no year) | `pl_all` |
| `tenant_query` | `by_tenant` |
| `comparison` + 2024 | `comparison_2024` |
| `comparison` + 2025 | `comparison_2025` |

### 7.3 Fuzzy Name Resolution (sync, microseconds)

Applied in `extractor` before the Polars query — normalises typos like "building17" → "Building 17":

```python
from difflib import get_close_matches

def fuzzy_resolve(value: str, candidates: set[str]) -> str | None:
    matches = get_close_matches(value, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None
```

**Retrieval latency:**

- Tier 1 cache hit: **0 ms**
- Tier 2 prefetch hit: **~0 ms**
- Tier 3 fresh query: **5–50 ms**

---

## 8. Layer 3 — Parallel Reasoning + Synthesis (`reason_and_synthesize`)

The biggest latency optimization in v2: **Critic+Judge (<1 ms) and Synthesizer (200–600 ms)
start simultaneously**. The synthesis task is cancelled only if the critic fails —
otherwise it has already been delivering tokens for ~200 ms by the time critic finishes.

```python
async def reason_and_synthesize(state: AgentState) -> dict:
    """Start synthesis immediately; cancel only if critic/judge fail."""
    critic_task = asyncio.create_task(_run_critic_and_judge(state))
    synth_task  = asyncio.create_task(_run_synthesis(state))

    updated = await critic_task  # completes in < 1 ms

    if not updated.get("judgment", True):
        synth_task.cancel()
        await asyncio.shield(asyncio.sleep(0))  # allow cancellation to propagate
        return {**updated, "final_response": updated["clarification_message"]}

    final_response = await synth_task
    return {**updated, "final_response": final_response}
```

**Layer 3 latency:**

- Critic + Judge: **< 1 ms**
- Synthesis first token: **~150 ms** (started in parallel with critic)
- Synthesis complete: **200–600 ms** (streaming to UI throughout)

---

## 9. LangGraph Async Graph Definition

```python
from langgraph.graph import StateGraph, END

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("extractor",         extractor)           # Layer 1
    graph.add_node("retrieval",         retrieval)           # Layer 2
    graph.add_node("reason_and_synth",  reason_and_synthesize)  # Layer 3
    graph.add_node("clarify",           _clarify_node)

    graph.set_entry_point("extractor")
    graph.add_edge("extractor", "retrieval")
    graph.add_edge("retrieval", "reason_and_synth")
    graph.add_conditional_edges(
        "reason_and_synth",
        lambda s: "clarify" if s.needs_clarification else END,
        {"clarify": "clarify", END: END},
    )
    graph.add_edge("clarify", END)
    return graph.compile()
```

### Execution Timeline (happy path, prefetch hit)

```text
t=0ms    ┌──────────────────────────────────────────────────┐
         │ extractor                                        │
         │  [thread A] LLM: parse intent + filters         │
         │  [thread B] Polars: 7 aggregations in parallel  │ ← asyncio.gather
t=~200ms └──────────────────────────────────────────────────┘
         ┌──────────────────────────────────────────────────┐
         │ retrieval: prefetch hit → ~0 ms                 │
t=~201ms └──────────────────────────────────────────────────┘
         ← Streamlit renders DATA TABLE here (raw_results ready)
         ┌──────────────────────────────────────────────────┐
         │ reason_and_synthesize                            │
         │  [coroutine A] Critic+Judge → < 1 ms           │
         │  [thread B]    LLM synthesis starts             │ ← asyncio.gather
         │  first token → UI queue at ~150 ms              │
t=~550ms └──────────────────────────────────────────────────┘
         ← Full prose streamed to UI; table already visible since t=201ms

Total perceived latency:
  Data table: ~200 ms
  First prose token: ~350 ms
  Full response: ~550–1,000 ms
  Cache hit: < 5 ms
```

---

## 10. Streamlit UI — Two-Phase Rendering

### 10.1 Features

- Chat history via `st.session_state`
- Lazy graph build on first query (startup stays fast)
- Auto Plotly bar chart for summary/comparison/tenant intents
- Expandable raw data table with per-row detail
- Per-query telemetry badge: `⚡ 412ms | 🟢 cache hit | intent: pl_summary`
- Cache hit indicator with entry count
- Node timing breakdown: `extractor: 380ms · retrieval: 0ms · reason_and_synthesize: 210ms`
- One-click example query buttons in the chat panel
- Instant KPI sidebar (P&L by year, properties count, tenants count) — pre-computed at startup

### 10.2 Sidebar

```python
def render_sidebar():
    st.sidebar.title("PropCo Portfolio")
    for year, profit in sorted(total_profit_by_year.items(), reverse=True):
        st.sidebar.metric(f"P&L {year}", f"€{profit:,.0f}")
    st.sidebar.metric("Properties", len(available_properties))
    st.sidebar.metric("Tenants", len(available_tenants))
```

---

## 11. Response Cache

```python
from cachetools import TTLCache
import hashlib, json

_CACHE: TTLCache = TTLCache(maxsize=256, ttl=300)  # 5-min TTL

def make_cache_key(intent: str, filters: dict) -> str:
    payload = json.dumps({"intent": intent, "filters": filters}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()
```

- Cache hit saves **350–1,000 ms** per repeated query (entire LLM pipeline skipped)
- TTL-based invalidation (5 min); explicit `cache_clear()` on data reload
- Synthesis prose is NOT cached (regeneration is cheap; caching risks stale wording)
- Scope: `src/cache.py` — `cache_get`, `cache_set`, `cache_clear`, `cache_size`

---

## 12. LLM Singleton + Async Thread Safety

`llama-cpp-python` is not thread-safe for concurrent inference. An `asyncio.Lock`
serialises all LLM calls within one process.

```python
_LLM: Llama | None = None
_LLM_LOCK: asyncio.Lock = asyncio.Lock()  # lazy-init inside event loop

async def llm_call_async(prompt: str, **kwargs) -> str:
    async with _LLM_LOCK:
        return await asyncio.to_thread(get_llm(), prompt, **kwargs)
```

LLM pre-warm at startup:

```python
async def prewarm_llm() -> None:
    """No-op inference to load model weights before first real query."""
    async with _LLM_LOCK:
        await asyncio.to_thread(lambda: get_llm()("Hi", max_tokens=1))
```

---

## 13. Docker Setup

### 13.1 Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential cmake libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /models
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(\
    repo_id='bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', \
    filename='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', \
    local_dir='/models'\
)"

COPY data/propco.csv data/
RUN python scripts/convert_to_parquet.py

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.runOnSave=false", "--server.fileWatcherType=none"]
```

### 13.2 docker-compose.yml

```yaml
version: "3.9"
services:
  propco-agent:
    build: .
    ports:
      - "8501:8501"
    environment:
      - N_GPU_LAYERS=0
      - N_THREADS=4
      - CACHE_TTL=300
      - CACHE_MAXSIZE=256
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
    deploy:
      resources:
        limits:
          memory: 8G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 13.3 GPU Variant

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
RUN pip install llama-cpp-python --force-reinstall --no-cache-dir
# With N_GPU_LAYERS=35:
#   Extractor: ~50–120 ms  (was 150–400 ms CPU)
#   Synthesizer: ~70–200 ms (was 200–600 ms CPU)
#   Total: ~150–350 ms
```

---

## 14. Performance Budget (v2)

| Phase | Component | Mechanism | Target Latency |
| --- | --- | --- | --- |
| **Startup** | DF load + LLM prewarm + aggregations | `asyncio.gather` at boot | One-time ~2–5 s |
| **Layer 1** | Extractor + Prefetch | Parallel `gather` | **150–400 ms** |
| **Layer 2** | Retrieval (cache/prefetch hit) | Dict lookup | **0 ms** |
| **Layer 2** | Retrieval (fresh query) | Polars in thread | **5–50 ms** |
| **Layer 3** | Critic + Judge | Pure Python | **< 1 ms** |
| **Layer 3** | Synthesis first token | Parallel with critic | **~150 ms** |
| **Layer 3** | Synthesis complete | Streaming | **200–600 ms** |
| **UX** | **Data table visible** | After retrieval node | **~200 ms** |
| **UX** | **First prose token** | Streaming | **~350 ms** |
| **UX** | **Full response** | Streaming complete | **~550–1,000 ms** |
| **Cache hit** | **Entire pipeline** | TTLCache lookup | **< 5 ms** |

---

## 15. Project Structure

```text
propco-agent/
├── ui/streamlit_app.py             # Entry point + two-phase async chat renderer
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── cortex.parquet
├── data.md
├── implementation.md
├── src/
│   ├── cache.py                    # TTLCache singleton + make_cache_key()
│   ├── data/
│   │   ├── loader.py               # Parquet loader + 8 startup caches (warm_up)
│   │   └── fuzzy.py                # difflib fuzzy resolver + resolve_filters()
│   ├── llm/
│   │   └── local.py                # Llama singleton + asyncio.Lock + prewarm_llm()
│   ├── agents/
│   │   ├── extractor.py            # async extractor() — LLM ‖ speculative prefetch
│   │   ├── retrieval.py            # async retrieval() — 3-tier cache
│   │   └── reason_and_synthesize.py # async: Critic+Judge ‖ Synthesizer
│   ├── graph/
│   │   ├── state.py                # AgentState Pydantic BaseModel
│   │   └── workflow.py             # build_graph() — 3-node async pipeline
│   └── utils/
│       ├── config.py               # Pydantic v2 Settings
│       └── llm_client.py           # OpenAI AsyncClient
├── scripts/
│   └── preprocess_data.py
└── tests/
    ├── conftest.py
    ├── test_extraction.py
    ├── test_retrieval.py (test_query_tools.py)
    ├── test_graph.py
    ├── test_cache.py
    ├── test_matching.py
    ├── test_executor.py (data loader tests)
    ├── test_memory.py (data accuracy tests)
    ├── test_planner.py (synthesizer tests)
    ├── test_calculations.py (retrieval accuracy tests)
    └── test_e2e.py
```

---

## 16. Requirements

```text
langgraph>=0.2.0
langchain-core>=0.2.0
llama-cpp-python>=0.2.90
polars>=0.20.0
streamlit>=1.35.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
cachetools>=5.3.0
plotly>=5.20.0
openai>=1.12.0
httpx>=0.27.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

---

## 17. Error Handling Matrix

| Condition | Handler | User message |
| --- | --- | --- |
| Property not in dataset | Critic flags; `reason_and_synthesize` cancels synth | "Property X not found. Available: [list]" |
| Empty result set | Critic flags; Judge decides by iteration count | "No records matched your filters." |
| JSON parse failure in extractor | Retry once at temp=0; fallback to `general` | Transparent |
| LLM timeout | `asyncio.wait_for` timeout; returns raw table only | Data table shown, no prose |
| Prefetch returns empty | Retrieval falls through to fresh Polars query | Transparent |
| Concurrent LLM contention | `asyncio.Lock` serializes; requests queue | Slight added wait; no crash |
| Cache stale | TTL expires in 5 min; or `cache_clear()` on data reload | Transparent after expiry |
| Synthesis cancelled mid-stream | `task.cancel()` + clarification returned instead | Clarification message shown |

---

## 18. Async Anti-Patterns Explicitly Avoided

| Anti-pattern | How it's avoided |
| --- | --- |
| `time.sleep()` in async code | All waits use `await asyncio.sleep()` or `asyncio.to_thread` |
| Blocking I/O on the event loop | All Polars + LLM calls wrapped in `asyncio.to_thread` |
| Sequential `await` for independent tasks | Always `asyncio.gather` or `asyncio.create_task` |
| Shared mutable state across tasks | `_DF` and caches are read-only after startup |
| LLM concurrent access without lock | `asyncio.Lock` serializes all LLM calls |
| Cold-start model download | GGUF baked into Docker image at build time |
| Re-loading Parquet per request | Single `_DF` loaded at startup, memory-resident |
| Synthesizer waiting for critic | Both start simultaneously; synth cancelled only if needed |
| Prefetch results thrown away | Mapped via `_PREFETCH_KEY_MAP` and consumed by retrieval tier 2 |
| Startup blocking the UI | `warm_up()` called at Streamlit startup before first query |

---

## 19. Sample Queries and Expected Async Behaviour

| Query | Intent | Parallel ops | Cache after first? |
| --- | --- | --- | --- |
| "Total P&L 2024?" | `pl_summary` | Extractor ‖ prefetch (7 tasks) | Yes |
| "Compare Building 17 vs 120" | `comparison` | Extractor ‖ prefetch + 2 sub-queries | Yes |
| "Same query again" | `pl_summary` | Cache hit → 0 ms, no tasks | — |
| "Tenant 12 revenue Q2 2024?" | `tenant_query` | Extractor ‖ prefetch_by_tenant | Yes |
| "Bank charges this year?" | `ledger_query` | Extractor ‖ prefetch_expenses | Yes |
| "Nonsense xyz 123" | `general` | Extractor ‖ prefetch_all | No (no filters) |
