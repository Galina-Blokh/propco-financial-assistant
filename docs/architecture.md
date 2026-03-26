# Graph RAG Architecture

This document summarizes the system architecture described in `SPEC.md` and makes the main diagrams easier to find without searching through the full specification.

## Overview

The system is a LangGraph-based, multi-agent workflow for querying a Cortex real-estate ledger dataset with natural language.

Main architectural goals:
- keep data retrieval deterministic and Polars-backed
- keep reasoning on the backend side
- avoid aggregation-level mistakes and double-counting
- support follow-up questions through conversation memory
- make the pipeline observable through reasoning traces and metrics

Core layers:
- Presentation: `ui/streamlit_app.py` today, FastAPI in the production path
- Orchestration: LangGraph workflow in `src/graph/workflow.py`
- Agents: router, planner, executor, analyst, critic, judge
- Data access: Polars-backed query tools over `cortex.parquet`
- Configuration: `.env`-driven runtime settings via `src/utils/config.py`

## High-Level System Architecture

```mermaid
flowchart TB
    subgraph External["External Services"]
        OAI["OpenAI API\ngpt-4o / gpt-4o-mini"]
        MLF["MLflow Tracking\nSQLite backend"]
    end

    subgraph UI["Presentation Layer"]
        ST["Streamlit UI\n(Demo)"]
        FA["FastAPI\n(Production)"]
    end

    subgraph Core["LangGraph Orchestration"]
        direction TB
        MEM["Conversation Memory\n(resolve follow-ups)"]
        RT["Router / Extractor"]
        VG["Validation Gate"]
        PL["Planner Agent"]
        EX["Executor Agent\n(ReAct loop)"]
        AN["Analyst Agent"]
        CR["Critic Agent"]
        JG["Judge Agent"]
        FO["Format Output"]
    end

    subgraph Data["Data Layer"]
        PQ[("cortex.parquet")]
        CA[("Response Cache")]
    end

    subgraph Tools["Polars Tool Registry"]
        QR["query_revenue"]
        QE["query_expenses"]
        QP["query_pnl"]
        QT["query_tenant"]
        QTR["query_trend"]
        QS["query_portfolio_stats"]
        BM["Building Matcher"]
        TM["Tenant Matcher"]
        TP["Timeframe Parser"]
    end

    ST & FA --> MEM
    MEM --> RT
    RT --> OAI
    RT --> VG
    VG --> BM & TM & TP
    VG --> PL
    PL --> OAI
    PL --> EX
    EX --> QR & QE & QP & QT & QTR & QS
    QR & QE & QP & QT & QTR & QS --> PQ
    EX --> AN
    AN --> OAI
    AN --> CR
    CR --> OAI
    CR --> JG
    JG --> OAI
    JG --> FO
    FO --> CA
    FO --> MEM
    FO --> ST & FA

    RT & PL & EX & AN & CR & JG -.-> MLF
```

## Agent Flow

```mermaid
stateDiagram-v2
    [*] --> ResolveFollowUp
    ResolveFollowUp --> Router

    Router --> ValidationGate
    ValidationGate --> Clarification : confidence < threshold
    ValidationGate --> Planner : complex query
    ValidationGate --> Executor : simple query fast path

    Clarification --> FormatOutput

    Planner --> Executor
    Executor --> Analyst
    Analyst --> Critic

    Critic --> Judge : PASS
    Critic --> Planner : FAIL and retry available
    Critic --> FormatOutput : FAIL and retries exhausted

    Judge --> FormatOutput
    FormatOutput --> UpdateMemory
    UpdateMemory --> [*]
```

## Data Hierarchy

The Cortex dataset is hierarchical and queries must respect aggregation boundaries:

- Entity level: `property_name IS NULL AND tenant_name IS NULL`
- Property level: `property_name = X AND tenant_name IS NULL`
- Tenant level: `property_name = X AND tenant_name = Y`

This distinction is critical because property questions must not accidentally aggregate tenant rows unless the user explicitly asks for tenant analysis.

```mermaid
graph TD
    E["PropCo\nentity level"]
    E --> B17["Building 17"]
    E --> B120["Building 120"]
    E --> B140["Building 140"]
    E --> B160["Building 160"]
    E --> B180["Building 180"]

    B17 --> T17P["Property rows\ntenant = NULL"]
    B17 --> T4["Tenant 4"]
    B17 --> T6["Tenant 6"]

    B120 --> T120P["Property rows\ntenant = NULL"]
    B120 --> T8["Tenant 8"]
    B120 --> T10["Tenant 10"]
    B120 --> T12["Tenant 12"]
```

## Backend Reasoning Ownership

Reasoning is a backend concern.

Rules:
- `reasoning_trace` is created in backend workflow state
- the UI may render the trace but must not invent or reconstruct it
- each major node can append trace entries for debugging and observability

Relevant code:
- `src/graph/workflow.py`
- `src/graph/state.py`
- `src/agents/`

## Current Implementation Notes

The current implementation already follows these main directions:
- `.env` is the source of truth for runtime settings
- Python 3.12 is the supported environment baseline
- query tools use Polars over the parquet dataset
- tests validate routing, memory, planner behavior, and tool correctness

For delivery sequencing and rollout checklists, see `docs/implementation_plan.md`.
