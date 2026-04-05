"""Layer 3b — Synthesizer: LLM response generation with real token streaming.

Runs in parallel with Critic via LangGraph fan-out edges.
If a ``token_queue`` is present in state, tokens are pushed as they arrive
from the LLM for live UI updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from src.llm.local import chat_complete, chat_complete_stream

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM = """\
You are a concise real estate financial assistant for PropCo.
PropCo owns 5 commercial buildings (17, 120, 140, 160, 180) with 18 tenants.
All financial data is from the Cortex ledger system. Amounts are in EUR.

Instructions:
1. Use exact EUR amounts (format: €X,XXX.XX)
2. Cite the data period when relevant (e.g. "In Q2 2024…")
3. Flag anomalies or unusual patterns
4. Never invent data — only use what is provided in the observations
5. If data is incomplete, clearly state what is missing
6. Maximum 3 short paragraphs; end with a one-sentence summary"""

GENERAL_SYSTEM = """\
You are a helpful assistant. Answer the user's question concisely and accurately
using your general knowledge. If you don't know, say so clearly."""

SYNTHESIS_USER = """\
Query: {user_query}
Intent: {intent}
Filters applied: {filters}
Data:
{raw_results}

Response:"""

SYNTHESIS_USER_WITH_HISTORY = """\
Prior conversation:
{history}

Query: {user_query}
Intent: {intent}
Filters applied: {filters}
Data:
{raw_results}

Response:"""


async def synthesizer(state: dict) -> dict:
    """Generate a natural-language response from retrieval data."""
    t0 = time.perf_counter()
    user_query = state.get("user_query", "")
    intent = state.get("intent", "general")
    filters = state.get("filters") or {}
    raw_results = state.get("raw_results") or []
    queue: asyncio.Queue | None = state.get("token_queue")
    conversation_history: list[dict] = state.get("conversation_history") or []

    history_text = ""
    if conversation_history:
        history_text = "\n".join(
            f'{m["role"]}: {m["content"]}' for m in conversation_history[-4:]
        )

    if intent == "general" and not raw_results and not filters:
        if history_text:
            user_msg = f"Prior conversation:\n{history_text}\n\nCurrent question: {user_query}"
        else:
            user_msg = user_query
        response = await _stream_or_complete(
            system=GENERAL_SYSTEM,
            user=user_msg,
            max_tokens=600,
            temperature=0.3,
            queue=queue,
        )
    else:
        data_str = (
            json.dumps(raw_results[:15], default=str, separators=(",", ":"))
            if raw_results
            else "No data retrieved."
        )
        if history_text:
            user_msg = SYNTHESIS_USER_WITH_HISTORY.format(
                history=history_text,
                user_query=user_query,
                intent=intent,
                filters=json.dumps(filters, default=str),
                raw_results=data_str,
            )
        else:
            user_msg = SYNTHESIS_USER.format(
                user_query=user_query,
                intent=intent,
                filters=json.dumps(filters, default=str),
                raw_results=data_str,
            )
        response = await _stream_or_complete(
            system=SYNTHESIS_SYSTEM,
            user=user_msg,
            max_tokens=600,
            temperature=0.2,
            queue=queue,
        )

    return {
        "final_response": response,
        "timings": {"synthesizer": time.perf_counter() - t0},
    }


async def _stream_or_complete(
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    queue: asyncio.Queue | None,
) -> str:
    """Use true streaming when a token_queue is available, else non-streaming."""
    if queue:
        collected: list[str] = []
        async for token in chat_complete_stream(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            collected.append(token)
            await queue.put(token)
        await queue.put(None)
        return "".join(collected)

    return await chat_complete(
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )
