"""Event-loop-aware AsyncOpenAI client factory.

Each ``asyncio.run()`` call in Streamlit creates (and later destroys) a fresh
event loop.  A module-level ``httpx.AsyncClient`` would bind its connection
pool to the *first* loop and silently break on subsequent loops.  The factory
below caches one ``AsyncOpenAI`` instance per event-loop identity so the
connection pool is always valid.
"""

from __future__ import annotations

import asyncio

import httpx
from openai import AsyncOpenAI

from src.utils.config import settings

_client: AsyncOpenAI | None = None
_client_loop_id: int | None = None


def get_llm_client() -> AsyncOpenAI:
    """Return an ``AsyncOpenAI`` client bound to the *current* event loop."""
    global _client, _client_loop_id
    loop = asyncio.get_running_loop()
    lid = id(loop)
    if _client is None or _client_loop_id != lid:
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        _client_loop_id = lid
    return _client
