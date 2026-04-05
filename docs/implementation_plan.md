# Implementation plan

This document tracks **delivery milestones**, **deployment checklists**, and **optional follow-up work**.  
For the authoritative system design, behavior, and APIs, see [`SPEC.md`](../SPEC.md) in the repo root.

---

## Current status (v3)

The following are **implemented** in code and reflected in `SPEC.md`:

| Area | Status |
|------|--------|
| LangGraph v3 — extractor ∥ prefetcher, critic ∥ synthesizer | Done |
| `AgentState` with merge reducer, `conversation_history` | Done |
| 3-tier retriever (response cache, prefetch, Polars) | Done |
| Extractor ambiguity / early clarify path | Done |
| OpenAI: extraction vs reasoning models + true streaming | Done |
| `get_llm_client()` per event loop (Streamlit-safe) | Done |
| Streamlit: input top, active Q&A, reverse-chron history, `on_click` examples | Done |
| MLflow logging (params, metrics, prompt/response artifacts) | Done |

---

## Milestones (reference)

| Phase | Focus | Notes |
|-------|--------|--------|
| M1 | Data + Polars loader, caches | `cortex.parquet`, `warm_up()` |
| M2 | Graph + agents | Six nodes + gate + clarify |
| M3 | UI + streaming | `token_queue`, telemetry under answers |
| M4 | Hardening | Multi-turn memory, ambiguity, resilient UI session state |
| M5 | Ops | MLflow URI, `.gitignore` for `mlruns/` / `mlruns.db` |

Future phases are optional backlog items below, not commitments.

---

## Optional backlog (not in SPEC as requirements)

Prioritize based on product needs:

1. **Latency** — Further reduce time-to-first-token (prompt trimming, caching extraction, optional skip prefetch for `general` intent).
2. **MLflow** — Structured eval metrics / golden-question runs; compare model versions over time.
3. **API surface** — Harden or document FastAPI entrypoint if exposing the graph beyond Streamlit.
4. **Local LLM** — Document hardware expectations; tune `n_ctx` / threads for your machine.

---

## Deployment checklist

Use this when shipping or handing off an environment.

- [ ] Copy `.env.example` → `.env`; set `OPENAI_API_KEY` (and models if non-default).
- [ ] `CORTEX_DATA_PATH` points to `cortex.parquet` (or your ledger file).
- [ ] `uv sync` (Python 3.12).
- [ ] Run tests: `uv run pytest tests/ --ignore=tests/test_e2e.py -v`.
- [ ] Smoke: `bash scripts/run_demo.sh` (Streamlit + MLflow) or Streamlit alone per README.
- [ ] MLflow URI in `.env` matches the UI (`sqlite:///mlruns.db` by default).
- [ ] Do not commit `.env`, `mlruns/`, or `mlruns.db`.

---

## Related documents

| File | Role |
|------|------|
| [`SPEC.md`](../SPEC.md) | Full technical specification (v3) |
| [`data.md`](../data.md) | Dataset schema and statistics |
| [`decisions.md`](decisions.md) | Architecture decision records (ADRs) |
