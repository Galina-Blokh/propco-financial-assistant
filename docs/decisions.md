# Architecture Decisions

This file records key architectural decisions for the project in a lightweight ADR-style format.

## ADR-001: Use LangGraph for orchestration

Status: accepted

Context:
- The system needs a multi-step workflow with routing, planning, execution, critique, and scoring.
- We also need conditional transitions, retries, and shared workflow state.

Decision:
- Use LangGraph as the orchestration layer for the agent pipeline.

Consequences:
- The workflow is explicit and testable.
- State transitions are easier to reason about than a purely prompt-driven chain.
- The project becomes coupled to LangGraph concepts such as nodes, edges, and shared state.

## ADR-002: Use Polars over the parquet dataset

Status: accepted

Context:
- The project works on a structured financial ledger dataset.
- Performance and deterministic aggregation matter more than flexible notebook-style manipulation.

Decision:
- Use Polars for all retrieval and aggregation logic over `cortex.parquet`.

Consequences:
- Queries are fast and consistent.
- Aggregation logic is transparent and testable.
- The application can avoid asking LLMs to do raw financial math over unstructured text.

## ADR-003: Keep reasoning on the backend

Status: accepted

Context:
- The system exposes a reasoning trace for observability and debugging.
- The UI should not invent or reconstruct reasoning because that creates trust and consistency problems.

Decision:
- Generate `reasoning_trace` in backend workflow state only.
- The UI may display the trace, but must render backend-provided data.

Consequences:
- Reasoning remains auditable and consistent across interfaces.
- UI complexity stays lower.
- Backend response payloads must carry enough trace information for inspection.

## ADR-004: Use `.env` as the runtime configuration source of truth

Status: accepted

Context:
- The project needs predictable runtime configuration across local development and test environments.
- Silent Python-side defaults caused confusion and drift.

Decision:
- Use `.env`-driven configuration via `pydantic-settings`.
- Require all runtime settings explicitly and fail fast when required values are missing.

Consequences:
- Startup fails early when configuration is incomplete.
- Local setup is more predictable.
- Tests must explicitly provide required settings.

## ADR-005: Standardize on Python 3.12

Status: accepted

Context:
- A release-candidate Python runtime surfaced async dependency warnings during test runs.
- The project successfully validated on a stable Python 3.12 environment.

Decision:
- Standardize local development and environment bootstrap on Python 3.12.

Consequences:
- Async test runs are more stable and less noisy.
- Tooling, docs, and environment scripts need to stay aligned with Python 3.12.
- Contributors using older Python versions may need to recreate their virtual environments.

## ADR-006: Separate specification from implementation planning

Status: accepted

Context:
- `SPEC.md` mixed behavior requirements with delivery sequencing, checklists, and milestones.
- That made the spec harder to scan as a source of truth for system behavior.

Decision:
- Keep architecture and behavior in `SPEC.md`.
- Move execution-oriented planning to `docs/implementation_plan.md`.

Consequences:
- The specification is easier to use as a requirements document.
- Delivery planning remains available without cluttering the core spec.
