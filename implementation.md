# Implementation Guide (v2)

## Architecture Overview

Three async LangGraph nodes replace the previous 6-agent, 8-LLM-call architecture:

```
Node 1 — extractor          ~150–400 ms   (1 LLM call + parallel Polars prefetch)
Node 2 — retrieval          <5 ms         (3-tier cache: TTL → prefetch → Polars)
Node 3 — reason_and_synth   ~200–600 ms   (critic+judge Python ║ synthesizer LLM)
──────────────────────────────────────────────────────────────────────
Total: 2 LLM calls · ~350–1,100 ms uncached · <5 ms cache hit
```

---

## Running the App

### OpenAI (default)

```bash
cp .env.example .env        # set OPENAI_API_KEY
PYTHONPATH=. uv run streamlit run ui/streamlit_app.py --server.headless=true
```

### Local GGUF (no API cost)

```bash
# Install llama-cpp-python (CPU)
uv add llama-cpp-python

# GPU (CUDA)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" uv add llama-cpp-python --force-reinstall

# Download model (~4.7 GB)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir /models

# .env settings
LLM_BACKEND=local
LOCAL_MODEL_PATH=/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
LOCAL_N_GPU_LAYERS=35
LOCAL_N_THREADS=4
```

---

## Testing

```bash
# Unit tests (excludes e2e)
PYTHONPATH=. uv run pytest tests/ --ignore=tests/test_e2e.py -v

# Smoke test — data layer
PYTHONPATH=. uv run python -c "
from src.data.loader import warm_up, get_cache
warm_up()
print('Properties:', sorted(get_cache('available_properties')))
print('Profit 2024: €', round(get_cache('total_profit_by_year').get('2024', 0), 2))
"

# Smoke test — retrieval
PYTHONPATH=. uv run python -c "
from src.agents.retrieval import retrieval_sync
r = retrieval_sync({'intent': 'comparison', 'filters': {'year': '2024'}})
print(r['raw_results'][:2])
"
```

---

## Startup Caches (`warm_up()`)

| Key | Content |
| --- | --- |
| `available_properties` | `{'Building 17', 'Building 120', ...}` |
| `available_tenants` | `{'Tenant 4', ...}` |
| `available_years` | `['2024', '2025']` |
| `available_quarters` | `['2024-Q1', ...]` |
| `available_ledger_categories` | `{'bank_charges', 'rental_income', ...}` |
| `available_ledger_groups` | `{'general_expenses', 'rental_income', ...}` |
| `total_profit_by_year` | `{'2024': 1171521.55, ...}` |
| `total_profit_by_property` | `{'Building 17': ..., ...}` |
| `total_profit_by_tenant` | `{'Tenant 4': ..., ...}` |
| `total_profit_by_quarter` | `{'2024-Q1': ..., ...}` |

---

## Extending the System

### Add a new query intent

1. Add the intent name to `SYSTEM_PROMPT` in `src/agents/extractor.py`
2. Add a routing branch in `_route_intent()` in `src/agents/retrieval.py`

### Add a new filter field

1. Add the field to `filter_col_map` in `_apply_filters()` (`retrieval.py`)
2. If it needs fuzzy/normalization matching, extend `resolve_filters()` in `src/data/fuzzy.py`
3. Expose it in the extractor `SYSTEM_PROMPT`

### Switch LLM backend

```bash
# .env
LLM_BACKEND=local
LOCAL_MODEL_PATH=/path/to/model.gguf
```

`src/llm/local.py` routes automatically. Uses `asyncio.Lock` to serialize calls for thread-safe GGUF inference.

### Update data

Re-run `scripts/preprocess_data.py` after modifying `cortex.parquet`.

---

## Key Design Decisions

### Merged Critic + Judge into `reason_and_synthesize`

Both were deterministic Python (<1 ms). Keeping them as separate graph nodes added routing overhead with zero benefit. Merged into one node that runs critic+judge first, then fires the synthesizer LLM with `asyncio.create_task`.

### Speculative prefetch

While the LLM parses the user query (~150–400 ms), Node 1 simultaneously runs 7 common Polars aggregations. Node 2 checks if any result matches the extracted filters — ~85 % hit rate means retrieval costs 0 ms for most queries.

### TTLCache (`src/cache.py`)

`cachetools.TTLCache(maxsize=256, ttl=300)` keyed on `(intent, sorted_filters)`. Identical repeated queries (e.g. dashboard refreshes) skip the entire pipeline in <5 ms.

### `@st.cache_resource` for the graph

The compiled `StateGraph` (with Pydantic schema validation) is expensive to build. `@st.cache_resource` ensures it is compiled exactly once per server process, surviving all Streamlit reruns including Clear Chat.
