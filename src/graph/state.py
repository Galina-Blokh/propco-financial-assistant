"""AgentState Pydantic model — the shared state passed through all LangGraph nodes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentState(BaseModel):
    # Input
    user_query: str

    # Layer 1 — Extractor output
    intent: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)

    # Speculative prefetch (populated in parallel with extraction)
    prefetch_results: dict[str, list[dict]] = Field(default_factory=dict)

    # Layer 2 — Retrieval output
    raw_results: list[dict[str, Any]] | None = None
    retrieval_error: str | None = None
    cache_hit: bool = False

    # Layer 3 — Reasoning + Synthesis output
    critique: str | None = None
    judgment: bool = True
    needs_clarification: bool = False
    clarification_message: str | None = None
    final_response: str | None = None

    # Streaming — asyncio.Queue passed by reference; excluded from LangGraph serialization
    token_queue: Any = Field(default=None, exclude=True)

    # Control flow
    iterations: int = 0
    error: str | None = None

    # Telemetry
    timings: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
