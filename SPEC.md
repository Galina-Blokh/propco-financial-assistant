# PropCo — Async Parallel Multi-Agent System
## Specification v3

---

## 1. Overview

A **fully async, parallel multi-agent** real estate asset management assistant built with
LangGraph's `AsyncStateGraph`. Every agent node is `async def`. Independent work is
always `asyncio.gather`'d. The LLM backend is configurable (OpenAI API or local GGUF),
data is pre-loaded Polars Parquet, and the UI is Streamlit with two-phase streaming
rendering.

**Performance targets:**
| Metric | Target |
|---|---|
| Data table visible to user | ~200 ms |
| First prose token streamed | ~350 ms |
| Full response (fresh query) | ~550–1,000 ms |
| Cache hit (repeated query) | < 5 ms |
| LLM cold-start penalty | 0 ms (pre-warmed at startup) |

---

## 2. Multi-Agent Design with LangGraph

### 2.1 Agent Roster

The system has **six specialist agents**, each a LangGraph node. They are orchestrated
by a `StateGraph` with both sequential edges (where there is a true data dependency) and
parallel fan-out (where there is none).

| Agent | Node name | Type | Role |
|---|---|---|---|
| **Extractor** | `extractor` | Async LLM | Parses NL query → structured intent + filters |
| **Prefetcher** | `prefetcher` | Async Polars | Speculatively runs all common aggregations in parallel |
| **Retriever** | `retriever` | Async Polars | 3-tier cache lookup; fires targeted query on miss |
| **Critic** | `critic` | Async Python | Validates retrieval result; deterministic, no LLM |
| **Synthesizer** | `synthesizer` | Async LLM stream | Generates prose response; streams tokens to UI queue |
| **Gate** | `gate` | Async Python | Joins critic+synth; routes to clarify on hard failure |

### 2.2 Parallel Execution Strategy

LangGraph supports parallel node execution via **fan-out edges** — multiple nodes listed
as targets of a single source edge run concurrently. We use this in two places:

**Fan-out 1 — Extraction + Prefetch (Layer 1):**
```
[START] ──► [extractor]   ─┐
                            ├─► [join_layer1] ──► [retriever] ──► ...
[START] ──► [prefetcher]  ─┘
```
Both `extractor` and `prefetcher` receive the initial state and run simultaneously.
`join_layer1` merges their outputs before passing to `retriever`.

**Fan-out 2 — Critic + Synthesizer (Layer 3):**
```
[retriever] ──► [critic]      ─┐
                                ├─► [gate] ──► END
[retriever] ──► [synthesizer] ─┘
```
Both start immediately after retrieval. `gate` reads the critic verdict and either
passes the synthesis through, appends a caveat, or replaces it with clarification.

### 2.3 Full Graph Topology

```
                    ┌─────────────────────────────────────────────────────┐
                    │              LangGraph AsyncStateGraph               │
                    │                                                     │
  user_query ──► START                                                    │
                    │                                                     │
           ┌────────┴────────┐                                            │
           ▼                 ▼          ← Fan-out 1: parallel             │
      [extractor]       [prefetcher]                                      │
      (LLM: intent      (6 Polars                                         │
       + filters +       tasks in                                         │
       ambiguity)        gather)                                          │
           │                 │                                            │
           └────────┬────────┘                                            │
                    ▼                                                     │
             [join_layer1]                                                │
             merges intent,                                               │
             filters, prefetch                                            │
                    │                                                     │
              ┌─────┴─────┐         ← Conditional: ambiguity check       │
              ▼           ▼                                               │
         [retriever]   [clarify]→END   (if extractor flagged ambiguity)   │
              │                                                           │
     ┌────────┴────────┐                                                  │
     ▼                 ▼            ← Fan-out 2: parallel                 │
  [critic]       [synthesizer]                                            │
  (<1 ms,         (LLM → true                                             │
  pure Python)     token stream)                                          │
     │                 │                                                  │
     └────────┬────────┘                                                  │
              ▼                                                           │
            [gate]                                                        │
            if critic OK → END                                            │
            if critic FAIL → clarify → END                                │
              │                                                           │
             END                                                          │
              └─────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Orchestration | **LangGraph** `StateGraph` | Native async; fan-out edges for parallel nodes |
| LLM (extraction) | **OpenAI** `gpt-4o-mini` via `AsyncOpenAI` | Fast, cheap — extraction is a simple JSON task |
| LLM (synthesis) | **OpenAI** `gpt-4o` via `AsyncOpenAI` + `stream=True` | Higher quality prose; true token streaming |
| LLM (local) | **Llama-3.1-8B-Instruct Q4_K_M GGUF** via `llama-cpp-python` | Optional zero-latency local mode |
| LLM routing | `src/llm/local.py` `chat_complete()` / `chat_complete_stream()` | Unified async interface; per-call `model` param; routes by `LLM_BACKEND` env |
| Data engine | **Polars** + pre-loaded **Parquet** | SIMD columnar; 10–100× faster than pandas |
| Async runtime | **asyncio** — `gather`, `create_task`, `Queue` | Native; no extra deps |
| Response cache | **cachetools `TTLCache`** | O(1) hit; skips entire pipeline |
| UI | **Streamlit** + `asyncio.Queue` streaming | Token-by-token rendering; two-phase display |
| Config | **Pydantic v2 `BaseSettings`** | Type-safe env vars |
| Testing | **pytest** + **pytest-asyncio** | Full async coverage |

---

## 4. Shared State — `AgentState`

All agents read from and write to a single state dict. LangGraph passes a copy of the
state into each node; merged writes are applied after the node returns. Parallel nodes
write to **non-overlapping fields** — no custom reducer needed except for `timings`
(which uses a dict-merge reducer so both fan-out branches preserve their timing data).

```python
class AgentState(TypedDict, total=False):
    user_query: str
    conversation_history: list[dict[str, str]]  # last N turns from UI

    # Extractor (Layer 1a)
    intent: str
    filters: dict[str, Any]
    extraction_error: str | None

    # Prefetcher (Layer 1b)
    prefetch_results: dict[str, list[dict]]

    # Retriever (Layer 2)
    raw_results: list[dict] | None
    retrieval_error: str | None
    cache_hit: bool
    cache_tier: str | None   # "response_cache" | "prefetch" | "polars"

    # Critic (Layer 3a)
    critique_issues: list[str]
    critique_ok: bool

    # Synthesizer (Layer 3b)
    token_queue: Any          # asyncio.Queue — injected, never serialised
    final_response: str | None

    # Gate / control
    needs_clarification: bool
    clarification_message: str | None
    iterations: int

    # Telemetry
    timings: Annotated[dict[str, float], _merge_dicts]
```

`conversation_history` carries the last 3 turns (6 messages) from the Streamlit
session, enabling the extractor and synthesizer to resolve references like
"what about last year?" or "and Building 120?".

---

## 5. Startup — Pre-warm Everything Before First Query

```python
# In Streamlit UI (app.py):
# 1. warm_up() — loads Parquet, builds lookup caches (sync, fast)
# 2. prewarm_llm() — runs 1-token inference to load model weights
#    (background thread, non-blocking)
```

`warm_up()` is called inside `build_graph()` so the data is ready before the first
query. LLM pre-warm runs in a daemon thread and never blocks the UI.

---

## 6. Agent Implementations

### 6.1 Extractor (`src/agents/extractor.py`)

**Responsibility:** Parse user NL query → structured `intent` + `filters` via LLM.
**Runs in parallel with:** Prefetcher.
**Model:** `settings.openai_model_extraction` (gpt-4o-mini) — faster for simple JSON tasks.

- System prompt is built once and cached via `@functools.lru_cache`
- Includes `conversation_history` in the user message so the LLM resolves
  references ("what about last year?", "and Building 120?")
- LLM returns `confidence` (0–1) and optional `clarification_question`
- If confidence < 0.4 or LLM flags ambiguity → sets `needs_clarification` +
  `clarification_message` in state; pipeline short-circuits to clarify node
- Retry once at temp=0 on JSON parse failure
- Fallback to `general` intent on double failure
- Fuzzy-resolve property/tenant names via `difflib`

### 6.2 Prefetcher (`src/agents/prefetcher.py`)

**Responsibility:** Speculatively run 6 Polars aggregations in `asyncio.gather`.
**Runs in parallel with:** Extractor.

Covers: P&L by property (2024/2025), by tenant, comparison (2024/2025),
all properties. (`by_quarter` was removed — never referenced by the retriever's
`_PREFETCH_KEY_MAP`.)

### 6.3 Join Layer 1 (`src/agents/join_layer1.py`)

Synchronisation barrier. Logs merged results; no computation.

### 6.4 Retriever (`src/agents/retriever.py`)

**Responsibility:** Return `raw_results` using the fastest tier:
1. Response TTL cache → 0 ms
2. Prefetch hit → 0 ms
3. Fresh Polars query → 5–50 ms

### 6.5 Critic (`src/agents/critic.py`)

**Responsibility:** Validate retrieval. Pure Python, no LLM, < 1 ms.
**Runs in parallel with:** Synthesizer.

Checks: empty results, unknown properties/tenants/years, retrieval errors.

### 6.6 Synthesizer (`src/agents/synthesizer.py`)

**Responsibility:** Generate prose response with true OpenAI token streaming.
**Runs in parallel with:** Critic.

- Uses `chat_complete_stream()` (OpenAI `stream=True`) when a `token_queue` is
  present — real tokens are pushed to the UI as they arrive from the API
- Falls back to non-streaming `chat_complete()` when no queue is present
- Includes `conversation_history` in the prompt for coherent follow-up answers
- Compact data payload: max 15 rows of JSON with no indentation

### 6.7 Gate (`src/agents/gate.py`)

**Responsibility:** After both critic and synthesizer complete, decide:
- Hard fail (no data) → replace response with clarification
- Soft fail (partial data) → append caveat
- Pass → response stands

Uses the extractor's LLM-generated `clarification_question` when available
(e.g. "Did you mean Building 17 or Building 120?") instead of only the generic
"I couldn't find the data" message.

---

## 7. LangGraph Graph Builder

```python
# src/graph/workflow.py
g = StateGraph(AgentState)

g.add_edge(START, "extractor")
g.add_edge(START, "prefetcher")
g.add_edge("extractor", "join_layer1")
g.add_edge("prefetcher", "join_layer1")

# Conditional: skip retrieval if extractor flagged ambiguity
g.add_conditional_edges("join_layer1", _route_after_join,
                        {"clarify": "clarify", "retriever": "retriever"})

g.add_edge("retriever", "critic")
g.add_edge("retriever", "synthesizer")
g.add_edge("critic", "gate")
g.add_edge("synthesizer", "gate")
g.add_conditional_edges("gate", _route_gate, {"clarify": "clarify", END: END})
g.add_edge("clarify", END)
```

---

## 8. LLM Backend — Dual-Mode with Per-Call Model Routing

```python
# src/llm/local.py
async def chat_complete(system, user, max_tokens, temperature,
                        json_mode=False, model=None) -> str:
    """Non-streaming. `model` overrides default for OpenAI (ignored for local)."""
    ...

async def chat_complete_stream(system, user, max_tokens, temperature,
                               model=None) -> AsyncIterator[str]:
    """True OpenAI streaming (stream=True). Yields tokens as they arrive.
    For local backend: yields the full response at once (fallback)."""
    ...
```

- **Extractor** passes `model=settings.openai_model_extraction` (gpt-4o-mini)
- **Synthesizer** uses `chat_complete_stream()` for real token streaming to the UI
- Set `LLM_BACKEND=local` + `LOCAL_MODEL_PATH=/path/to.gguf` for zero-network-latency mode

---

## 9. Response Cache

```python
# src/cache.py — cachetools TTLCache
# Key: MD5(json({intent, filters}))
# Value: raw_results (list[dict])
# TTL: configurable via CACHE_TTL_SECONDS
```

Cache hit skips the entire pipeline except the extractor.

---

## 10. Streamlit UI — Async Two-Phase Rendering

**Phase 1:** Data table rendered from `raw_results` after graph completes.
**Phase 2:** Prose streamed token-by-token from `token_queue` via `asyncio.Queue`.

**Startup / white-screen avoidance:** The Streamlit module avoids importing
`src.graph.workflow` at import time. `build_graph()` runs inside
`@st.cache_resource _get_graph()` on the **first user query**, so the browser
can render the shell (sidebar, input, buttons) after only data `warm_up()`.
Eager LangGraph + agent imports previously blocked the first script run for
tens of seconds on some environments.

**Layout order:** Input + example buttons (with `on_click` callbacks) at top →
active Q&A streaming (newest) → chat history in reverse chronological order
(newest first, oldest at bottom).

**Multi-turn memory:** The UI passes the last 6 messages (3 turns) from
`st.session_state.messages` into `create_initial_state()` as
`conversation_history`. The extractor uses this to resolve references;
the synthesizer uses it for coherent follow-ups.

**Resilience:** After each turn, the user and assistant entries are appended
together (never leave a lone user message if the graph raises). Failures from
`asyncio.run()` are caught and surfaced as an assistant error line so the chat
stays paired and inputs keep working.

**History visibility:** `st.rerun()` aborts the rest of the script, so completed
turns must be drawn **inside** the `if prompt:` branch (below the live reply,
before append + rerun). MLflow logging runs in a **background thread** so SQLite
or slow IO never blocks `st.rerun()` (which previously made the app feel dead on
later turns).

```python
async def _run():
    token_queue = asyncio.Queue()
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[-6:]]
    initial = create_initial_state(user_query=prompt,
                                   conversation_history=history)
    initial["token_queue"] = token_queue
    graph_task = asyncio.create_task(graph.ainvoke(initial))

    # Drain queue — real tokens arrive from OpenAI stream=True
    while True:
        token = await asyncio.wait_for(token_queue.get(), timeout=0.05)
        if token is None: break
        prose_placeholder.markdown(streamed + "▌")

    result = await graph_task
```

---

## 11. Project Structure

```
propco-agent/
├── scripts/
│   └── run_demo.sh                 # Streamlit + MLflow UI (bash; same session)
├── ui/
│   └── streamlit_app.py            # Streamlit entry point
├── src/
│   ├── graph/
│   │   ├── state.py                # AgentState TypedDict + reducer
│   │   └── workflow.py             # build_graph() — fan-out topology
│   ├── agents/
│   │   ├── extractor.py            # Layer 1a — LLM intent parser
│   │   ├── prefetcher.py           # Layer 1b — parallel Polars prefetch
│   │   ├── join_layer1.py          # Layer 1 join barrier
│   │   ├── retriever.py            # Layer 2 — 3-tier cache retrieval
│   │   ├── critic.py               # Layer 3a — deterministic validator
│   │   ├── synthesizer.py          # Layer 3b — LLM prose + streaming
│   │   └── gate.py                 # Layer 3 join + conditional routing
│   ├── data/
│   │   ├── loader.py               # Parquet load + startup caches
│   │   └── fuzzy.py                # Fuzzy name matching (difflib)
│   ├── llm/
│   │   └── local.py                # Dual-mode LLM: OpenAI / local GGUF
│   ├── cache.py                    # TTLCache singleton
│   └── utils/
│       ├── config.py               # Pydantic v2 BaseSettings
│       └── llm_client.py           # get_llm_client() — per event loop (Streamlit + asyncio.run)
├── tests/
│   ├── conftest.py
│   ├── test_extraction.py          # Extractor unit tests
│   ├── test_graph.py               # Critic, gate, synthesizer, routing
│   ├── test_calculations.py        # Retriever accuracy vs real data
│   ├── test_cache.py               # TTLCache unit tests
│   ├── test_matching.py            # Fuzzy matching unit tests
│   └── test_e2e.py                 # Full pipeline (requires API key)
├── data/
│   └── cortex.parquet              # PropCo financial ledger
├── SPEC.md                         # This file
├── README.md
├── data.md                         # Cortex parquet schema + statistics
├── docs/
│   ├── decisions.md              # Architecture decision records (ADRs)
│   └── implementation_plan.md    # Milestones, deployment checklist, backlog
├── pyproject.toml
└── .env.example
```

---

## 12. Performance Budget

| Phase | What runs | How | Latency |
|---|---|---|---|
| **Startup** | Parquet load + cache build + LLM prewarm | sync + background thread | One-time 2–5 s |
| **Fan-out 1** | Extractor ‖ Prefetcher | Parallel LangGraph nodes | **150–400 ms** wall-clock |
| **Join 1** | Merge state | Near-zero | < 1 ms |
| **Retrieval** | Cache/prefetch hit | Dict lookup | **0 ms** |
| **Retrieval** | Fresh Polars query | `to_thread` | **5–50 ms** |
| **Fan-out 2** | Critic ‖ Synthesizer | Parallel LangGraph nodes | critic < 1 ms |
| **Synthesizer** | Full prose | LLM call | **200–600 ms** |
| **Gate** | Route decision | Pure Python | < 1 ms |
| **UI Phase 1** | Data table visible | After retriever | **~200 ms** |
| **UI Phase 2** | Prose streaming | Token queue | **~350 ms first token** |
| **Cache hit** | Entire pipeline | TTLCache | **< 5 ms** |

---

## 13. Error Handling Matrix

| Condition | Agent | Resolution |
|---|---|---|
| LLM JSON parse failure | Extractor | Retry once at temp=0 → fallback to `general` |
| Property name not found | Critic | Flags issue; gate clarifies if no data |
| Tenant name not found | Critic | Same as above |
| Empty retrieval result | Critic → Gate | Hard fail: gate replaces with clarification |
| Partial data with issues | Critic → Gate | Soft fail: gate appends ⚠️ caveat |
| LLM timeout | Synthesizer / httpx | Exception propagates; UI try/except shows error and still appends Q+A pair |
| Concurrent user requests | OpenAI per-loop client; local GGUF singleton | `asyncio.Lock` serialises local model calls only |
| Cache stale data | Cache | TTL expires; fresh query on next request |
| Ambiguous / vague query | Extractor | LLM returns low confidence + clarification question; pipeline short-circuits to clarify node |
| Fuzzy match fails | Extractor | Original string passed; critic catches mismatch |

---

## 14. Async Anti-Patterns Explicitly Avoided

| Anti-pattern | How avoided |
|---|---|
| `time.sleep()` in async code | Only `await asyncio.sleep()` if needed |
| Blocking I/O on event loop | All Polars + llama-cpp wrapped in `asyncio.to_thread` |
| Sequential await for independent work | Fan-out edges in LangGraph; `asyncio.gather` in prefetcher |
| Shared mutable state between parallel nodes | Parallel nodes write to non-overlapping `AgentState` fields |
| LLM concurrent access without serialisation | OpenAI: SDK + per-loop client; local: `asyncio.Lock` on GGUF singleton |
| Re-parsing Parquet per request | `_DF` loaded once at startup; memory-resident |
| Synthesis waiting for critic | Both start simultaneously via fan-out 2 |
| Prefetch results discarded | Retriever maps intent → prefetch key via `_PREFETCH_KEY_MAP` |
| UI blocked until full response | Two-phase rendering: prose from queue, data from results |

---

## 15. Configuration

All settings loaded from `.env` via Pydantic v2 `BaseSettings`.
See `.env.example` for the full list.

Key environment variables:
- `LLM_BACKEND` — `"openai"` (default) or `"local"`
- `OPENAI_API_KEY` — required when using OpenAI backend
- `OPENAI_MODEL_REASONING` — model for synthesis (default: `gpt-4o`)
- `OPENAI_MODEL_EXTRACTION` — model for extraction (default: `gpt-4o-mini`)
- `LOCAL_MODEL_PATH` — path to GGUF file when using local backend
- `CORTEX_DATA_PATH` — path to the Parquet data file
- `CACHE_TTL_SECONDS` — response cache TTL
