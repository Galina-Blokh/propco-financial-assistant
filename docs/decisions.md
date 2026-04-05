# Architecture Decisions

Key architectural decisions for the PropCo Financial Assistant (v3).

## ADR-001: Use LangGraph for orchestration

Status: accepted

Context:
- The system needs a multi-step workflow with parallel fan-out, routing, and shared state.

Decision:
- Use LangGraph `StateGraph` with fan-out edges for parallel node execution.

Consequences:
- The workflow topology is explicit and testable.
- Parallel execution (extractor ‖ prefetcher, critic ‖ synthesizer) is native to the graph.
- The project is coupled to LangGraph concepts (nodes, edges, `TypedDict` state).

## ADR-002: Use Polars over the parquet dataset

Status: accepted

Context:
- The project works on a structured financial ledger dataset.
- Performance and deterministic aggregation matter more than flexible manipulation.

Decision:
- Use Polars for all retrieval and aggregation logic over `cortex.parquet`.

Consequences:
- Queries are fast and consistent.
- Aggregation logic is transparent and testable.
- Avoids asking LLMs to do raw financial math.

## ADR-003: Dual LLM backend (OpenAI + local GGUF)

Status: accepted

Context:
- OpenAI is convenient for development; local GGUF eliminates API cost and network latency.

Decision:
- Unified `chat_complete()` in `src/llm/local.py` routes by `LLM_BACKEND` env var.
- `asyncio.Lock` serialises local model calls for thread safety.

Consequences:
- Switching backends requires only a `.env` change.
- Local mode needs `llama-cpp-python` installed separately.

## ADR-004: `.env` as runtime configuration source of truth

Status: accepted

Decision:
- Use `.env`-driven configuration via `pydantic-settings`.
- All runtime settings loaded from environment; fail fast when missing.

Consequences:
- Startup fails early on incomplete config.
- Tests must explicitly provide required settings in `conftest.py`.

## ADR-005: Standardize on Python 3.12

Status: accepted

Decision:
- Standardize on stable Python 3.12 for local development and CI.

Consequences:
- Async test runs are stable and warning-free.
- Contributors on older Python need to recreate their venvs.

## ADR-006: TypedDict state with merge reducer for parallel fan-out

Status: accepted

Context:
- LangGraph fan-out nodes write to the same state dict concurrently.
- The `timings` field is written by both branches of each fan-out.

Decision:
- Use `TypedDict` (not Pydantic `BaseModel`) for `AgentState`.
- Apply `Annotated[dict, _merge_dicts]` reducer on `timings` so both branches' data is preserved.

Consequences:
- Parallel state merging is safe and deterministic.
- No custom reducer needed for other fields (they are non-overlapping).

## ADR-007: Speculative prefetch covers ~85% of queries

Status: accepted (updated)

Context:
- While the extractor LLM parses intent (~150–400 ms), the data layer is idle.

Decision:
- Run 6 Polars aggregation slabs in `asyncio.gather` (prefetcher node) in parallel with extraction.
- Retriever checks prefetch results before running a fresh query.
- The `by_quarter` slab was removed — it was never referenced by the retriever's `_PREFETCH_KEY_MAP`.

Consequences:
- Most queries hit the prefetch cache at 0 ms retrieval cost.
- One fewer thread pool task per query; remaining 6 slabs cover all retriever key mappings.

## ADR-008: Multi-turn conversation memory via state

Status: accepted

Context:
- Users expect follow-up questions to resolve references ("what about last year?", "and Building 120?").
- The pipeline was completely stateless between turns.

Decision:
- Add `conversation_history: list[dict[str, str]]` to `AgentState`.
- UI passes the last 6 messages (3 turns) from `st.session_state.messages` into `create_initial_state()`.
- Extractor includes conversation history in the LLM prompt to resolve references.
- Synthesizer includes prior Q&A for coherent follow-up answers.

Consequences:
- Follow-up queries resolve correctly without explicit re-stating of context.
- Input token count for extractor/synthesizer increases slightly (~200–400 tokens for 3 turns).
- No server-side persistence — memory is per Streamlit browser session.

## ADR-009: Extractor-driven ambiguity detection and clarification

Status: accepted

Context:
- Vague queries ("show data", "revenue", "compare") produce empty or wrong results.
- Clarification was only triggered after retrieval failure (wasting LLM + retrieval time).

Decision:
- Extractor LLM returns `confidence` (0–1), `needs_clarification`, and `clarification_question`.
- If confidence < 0.4 or LLM flags ambiguity, pipeline short-circuits from `join_layer1` to `clarify` node.
- Gate uses the extractor's LLM-generated question when available.

Consequences:
- Ambiguous queries are caught early — no wasted retrieval/synthesis.
- The graph has a new conditional edge after `join_layer1` (`_route_after_join`).
- False positives (low confidence on clear queries) are mitigated by the 0.4 threshold.

## ADR-010: True OpenAI streaming + per-call model routing

Status: accepted

Context:
- Synthesizer awaited the full LLM response, then fake-split it into word chunks — perceived latency was high.
- Both extractor and synthesizer used the same expensive model (`gpt-4o`), even though extraction is a simple JSON task.

Decision:
- Add `chat_complete_stream()` async generator to `src/llm/local.py` that uses OpenAI `stream=True`.
- Synthesizer pushes real tokens to the UI queue as they arrive from the API.
- Add optional `model` parameter to `chat_complete()` and `_openai_complete()`.
- Extractor uses `settings.openai_model_extraction` (gpt-4o-mini); synthesizer uses `settings.openai_model_reasoning` (gpt-4o).

Consequences:
- Time-to-first-visible-token drops from ~2s to ~400ms.
- Extraction is 2–3x faster with the smaller model.
- Local backend falls back to non-streaming (yields full response at once).

## ADR-011: Event-loop-aware OpenAI client (`get_llm_client`)

Status: accepted

Context:
- Streamlit runs each script execution synchronously; the UI calls `asyncio.run()` per query, which creates and tears down a new event loop each time.
- A module-level `AsyncOpenAI` backed by a long-lived custom `httpx.AsyncClient` bound its connection pool to the *first* loop. Later loops reused a stale client → hangs, slow calls, or broken input after a few turns.
- The same pattern affects `asyncio.Lock` used for local GGUF serialization if the lock outlives its loop.

Decision:
- Replace the singleton with `get_llm_client()` in `src/utils/llm_client.py`, caching one `AsyncOpenAI` per `id(asyncio.get_running_loop())`.
- Use SDK-managed HTTP timeouts (`httpx.Timeout` via constructor) instead of a shared custom `AsyncClient` unless a future need requires it.
- Recreate `_get_llm_lock()` in `local.py` when the running loop identity changes.

Consequences:
- OpenAI calls remain stable across unlimited Streamlit reruns.
- Slightly more client construction on first use of each new loop (negligible vs network latency).
