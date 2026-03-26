"""Response cache — TTLCache keyed on (intent, filters). O(1) hit skips all LLM calls."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cachetools import TTLCache

from src.utils.config import settings

_CACHE: TTLCache = TTLCache(
    maxsize=int(getattr(settings, "cache_maxsize", 256)),
    ttl=int(getattr(settings, "cache_ttl_seconds", 300)),
)


def make_cache_key(intent: str | None, filters: dict[str, Any]) -> str:
    """Stable MD5 key from intent + sorted filters."""
    payload = json.dumps({"intent": intent, "filters": filters}, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


def cache_get(key: str) -> list[dict] | None:
    return _CACHE.get(key)


def cache_set(key: str, value: list[dict]) -> None:
    _CACHE[key] = value


def cache_clear() -> None:
    _CACHE.clear()


def cache_size() -> int:
    return len(_CACHE)
