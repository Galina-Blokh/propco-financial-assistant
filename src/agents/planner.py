"""Layer 1 — Planner: LLM-based intent extraction + parallel worker decomposition.

Replaces the extractor + prefetcher fan-out from v2. A single LLM call extracts
intent and filters (same prompt as extractor), then deterministic logic decomposes
the query into one or more WorkerAssignment dicts dispatched via Send().

Comparison queries with ≥2 properties → one worker per property (true parallelism).
All other queries → single worker with the full filter set.
"""

from __future__ import annotations

import functools
import json
import logging
import time

from src.data.fuzzy import resolve_filters
from src.data.loader import get_cache
from src.llm.local import chat_complete
from src.utils.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a query parser for a real estate financial database.
Extract intent and filters from the user query.
Reply ONLY with valid JSON, no markdown, no explanation.

Available intents: pl_summary | property_detail | tenant_query | comparison | ledger_query | general

Available filter keys:
  property_name   (str)  — single property
  property_names  (list) — for comparison intent
  tenant_name     (str)
  year            (str)  — "2024" or "2025"
  quarter         (str)  — e.g. "2024-Q2"
  month           (str)  — e.g. "2024-M06"
  ledger_group    (str)
  ledger_category (str)
  ledger_type     (str)  — "revenue" or "expenses"

Known properties: {available_properties}
Known tenants: {available_tenants}
Known years: {available_years}
Known quarters: {available_quarters}
Known ledger_categories (use EXACT snake_case values): {available_ledger_categories}
Known ledger_groups (use EXACT snake_case values): {available_ledger_groups}

If prior conversation is provided, resolve any references (e.g. "last year", \
"that building", "same tenant") using the conversation context.

If the query is ambiguous or too vague to determine the intent/filters reliably, \
set "confidence" to a low value (<0.5) and provide a "clarification_question" that \
would help narrow down the query.

Return format:
{{"intent": "<intent>", "filters": {{<key>: <value>, ...}}, \
"confidence": 0.0-1.0, "needs_clarification": false, "clarification_question": null}}"""

USER_TEMPLATE = 'User query: "{user_query}"\n\nJSON:'

USER_TEMPLATE_WITH_HISTORY = """\
Conversation so far:
{history}

Current query: "{user_query}"

JSON:"""


@functools.lru_cache(maxsize=1)
def _build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        available_properties=sorted(get_cache("available_properties") or []),
        available_tenants=sorted(get_cache("available_tenants") or []),
        available_years=get_cache("available_years") or [],
        available_quarters=get_cache("available_quarters") or [],
        available_ledger_categories=sorted(get_cache("available_ledger_categories") or []),
        available_ledger_groups=sorted(get_cache("available_ledger_groups") or []),
    )


async def planner(state: dict) -> dict:
    """Extract intent+filters via LLM, then decompose into parallel worker assignments."""
    t0 = time.perf_counter()
    user_query: str = state["user_query"]
    conversation_history: list[dict] = state.get("conversation_history") or []

    system = _build_system_prompt()
    if conversation_history:
        history_text = "\n".join(
            f'{m["role"]}: {m["content"]}' for m in conversation_history[-6:]
        )
        user_msg = USER_TEMPLATE_WITH_HISTORY.format(
            history=history_text, user_query=user_query
        )
    else:
        user_msg = USER_TEMPLATE.format(user_query=user_query)

    raw = await chat_complete(
        system=system,
        user=user_msg,
        max_tokens=256,
        temperature=0.0,
        json_mode=True,
        model=settings.openai_model_extraction,
    )
    parsed = _parse_json(raw)
    if parsed is None:
        raw2 = await chat_complete(
            system=system,
            user=user_msg,
            max_tokens=256,
            temperature=0.0,
            json_mode=True,
            model=settings.openai_model_extraction,
        )
        parsed = _parse_json(raw2)
    if parsed is None:
        logger.warning("Planner: JSON parse failed twice, falling back to general")
        parsed = {"intent": "general", "filters": {}}

    intent: str = parsed.get("intent", "general")
    raw_filters = parsed.get("filters", {})
    if not isinstance(raw_filters, dict):
        raw_filters = {}
    filters = resolve_filters(raw_filters)

    confidence = float(parsed.get("confidence", 0.8))
    needs_clarification = parsed.get("needs_clarification", False) or confidence < 0.4
    clarification_question = parsed.get("clarification_question")

    worker_assignments = _plan_workers(intent, filters, user_query)

    result: dict = {
        "intent": intent,
        "filters": filters,
        "worker_assignments": worker_assignments,
        "timings": {"planner": time.perf_counter() - t0},
    }

    if needs_clarification and clarification_question:
        result["needs_clarification"] = True
        result["clarification_message"] = clarification_question

    return result


def _plan_workers(intent: str, filters: dict, user_query: str) -> list[dict]:
    """Decompose into parallel worker assignments.

    Comparison with ≥2 properties → one assignment per property.
    All other cases → single assignment.
    """
    property_names = filters.get("property_names") or []
    if intent == "comparison" and len(property_names) >= 2:
        shared = {k: v for k, v in filters.items() if k != "property_names"}
        return [
            {
                "worker_type": "comparison_worker",
                "filters": {**shared, "property_name": prop},
                "question": f"P&L breakdown for property {prop}",
            }
            for prop in property_names
        ]

    return [
        {
            "worker_type": intent or "pl_summary",
            "filters": filters,
            "question": user_query,
        }
    ]


def _parse_json(raw: str) -> dict | None:
    try:
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return None
