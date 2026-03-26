# Graph RAG Implementation Plan

This document contains execution-oriented planning material moved out of `SPEC.md` so the specification can stay focused on system behavior and architecture.

Source sections migrated from `SPEC.md`:
- `3.8.7 Optimization Priority (Implementation Order)`
- `10. Production Readiness Checklist`
- `11. Timeline & Milestones`
- `14.7 Implementation Checklist`

## 1. Optimization Priority

| Priority | Technique | Impact |
| --- | --- | --- |
| **P0** (do first) | Polars LazyFrame + pre-computed aggs | Data retrieval: 50ms → 5ms |
| **P0** | Connection pooling | -50–100ms per LLM call |
| **P0** | Streaming tokens | Time-to-first-token: ~200ms |
| **P1** | Complexity-based skip (fast path) | Simple queries: 10s → <1s |
| **P1** | Query fingerprint cache | Repeated queries: <50ms |
| **P1** | KV cache (prompt structure) | -30–50% input processing time |
| **P2** | Semantic response cache | Similar queries: <50ms |
| **P2** | Local SLM (ONNX) | Router: 150ms → 20ms |
| **P2** | Parallel executor | Comparisons: 2× faster |
| **P3** (later) | Learned query router | Router: 150ms → 2ms |
| **P3** | Distilled Critic | Critic: 800ms → 50ms |

## 2. Challenges

### LLM Output vs Data Schema Mismatch

The extractor LLM outputs human-readable values (e.g. `"Bank charges"`) while the parquet stores snake_case (`bank_charges`). Fixed by:
- Normalising `ledger_category` / `ledger_group` in `resolve_filters()` (lowercase + spaces → underscores)
- Passing `Known ledger_categories` as exact values in the extractor system prompt
- Secondary fuzzy fallback in `_apply_filters()` via `difflib.get_close_matches`

### Streamlit + asyncio Conflict

Streamlit runs its own event loop. Calling `asyncio.run()` inside a Streamlit script causes `RuntimeError: This event loop is already running` on some setups. Fixed by keeping `asyncio.run()` scoped to the LangGraph `ainvoke` call and running LLM pre-warm in a daemon thread so it never blocks the Streamlit loop.

### Slow App Startup

The original code called `asyncio.run(prewarm_llm())` at module-level, blocking the entire Streamlit UI for 60+ seconds while the GGUF model loaded. Fixed by moving it to a background `threading.Thread(daemon=True)` so the UI is immediately responsive.

### Graph Recompilation on Every Rerun

`StateGraph(AgentState)` with a Pydantic model schema is expensive to compile. Storing it in `st.session_state` still re-checked on every `st.rerun()`. Fixed with `@st.cache_resource` — compiled exactly once per server process lifetime.

### Double `st.rerun()` on Clear Chat

Every Streamlit button click already triggers a script rerun. Adding an explicit `st.rerun()` inside the handler caused two full script executions per clear. Fixed by removing the redundant `st.rerun()` call.

### Pydantic v2 Deprecation

`class Config` inside a Pydantic `BaseModel` is deprecated in v2. Replaced with `model_config = ConfigDict(arbitrary_types_allowed=True)`.

### Fuzzy Matching False Positives

The default `difflib` cutoff of 0.6 caused `"Building 999"` to match `"Building 17"`. Raised to 0.80 for property/tenant resolution.

---

## 3. Production Readiness Checklist

- [ ] All unit tests passing (`pytest --cov >85%`)
- [ ] MLflow tracking configured and logging metrics
- [ ] Error handling for all 9 identified edge cases (Section 4)
- [ ] Building/tenant fuzzy matching with alias table
- [ ] Aggregation level auto-detection working correctly
- [ ] README with setup instructions
- [ ] Demo showing all 4 query types + edge cases
- [ ] Data retrieval latency <500ms P95
- [ ] End-to-end latency <10s P95
- [ ] Judge score >0.85 average
- [ ] FastAPI deployed with health check endpoint
- [ ] All prompts tested against 10+ example queries each
- [ ] No double-counting in any aggregation path

## 3. Timeline And Milestones

| Phase | Time | Deliverables |
|-------|------|--------------|
| **Phase 1: Setup** | 0.5h | Python env, dependencies, project structure, data validation |
| **Phase 2: Schemas & Matching** | 1.0h | Pydantic models, building/tenant alias tables, timeframe parser |
| **Phase 3: Data Layer** | 1.5h | Polars queries for all 4 intents, aggregation level logic, edge cases |
| **Phase 4: Router Agent** | 1.0h | SLM extraction, intent classification, validation gate |
| **Phase 5: LLM Agents** | 2.0h | Analyst, Critic, Judge agents with corrected prompts |
| **Phase 6: LangGraph** | 1.0h | StateGraph, node wiring, retry logic |
| **Phase 7: Evaluation** | 0.5h | MLflow setup, unit tests, known-value assertions |
| **Phase 8: UI + Docs** | 0.5h | Streamlit demo, README |

**Total: 8 hours**

## 4. Implementation Checklist

### 4.1 Code Changes

- [ ] Implement lazy graph loading in `ui/streamlit_app.py`
- [ ] Fix type conversion issues in query tools
- [ ] Preserve tenant-analysis intent for short follow-ups like `Building 120`
- [ ] Allow portfolio-wide tenant ranking without forced clarification
- [ ] Add caching decorators for expensive operations
- [ ] Optimize data loading with preprocessed parquet
- [ ] Update environment configuration

### 4.2 Infrastructure

- [ ] Create optimized `Dockerfile`
- [ ] Set up Redis for caching
- [ ] Configure `docker-compose.yml`
- [ ] Add health checks and monitoring
- [ ] Set up log aggregation

### 4.3 Performance Testing

- [ ] Benchmark startup time
- [ ] Measure query response times
- [ ] Test concurrent user load
- [ ] Validate memory usage
- [ ] Monitor cache effectiveness

### 4.4 Production Deployment

- [ ] Update `.env.example` with performance defaults
- [ ] Add performance section to README
- [ ] Create deployment documentation
- [ ] Set up monitoring dashboards
- [ ] Configure alerting thresholds
