"""Tests for TTLCache response cache (src/cache.py)."""

from __future__ import annotations


def test_cache_set_and_get():
    from src.cache import cache_clear, cache_get, cache_set, make_cache_key
    cache_clear()
    key = make_cache_key("pl_summary", {"year": "2024"})
    assert cache_get(key) is None
    cache_set(key, [{"property_name": "Building 17", "total_profit": 100.0}])
    result = cache_get(key)
    assert result is not None
    assert result[0]["property_name"] == "Building 17"


def test_cache_key_is_stable():
    from src.cache import make_cache_key
    k1 = make_cache_key("comparison", {"year": "2024", "property_name": "Building 17"})
    k2 = make_cache_key("comparison", {"property_name": "Building 17", "year": "2024"})
    assert k1 == k2


def test_cache_key_differs_by_intent():
    from src.cache import make_cache_key
    k1 = make_cache_key("pl_summary", {"year": "2024"})
    k2 = make_cache_key("comparison", {"year": "2024"})
    assert k1 != k2


def test_cache_key_differs_by_filters():
    from src.cache import make_cache_key
    k1 = make_cache_key("pl_summary", {"year": "2024"})
    k2 = make_cache_key("pl_summary", {"year": "2025"})
    assert k1 != k2


def test_cache_clear():
    from src.cache import cache_clear, cache_get, cache_set, cache_size, make_cache_key
    cache_clear()
    cache_set(make_cache_key("general", {}), [{"profit": 1}])
    assert cache_size() >= 1
    cache_clear()
    assert cache_size() == 0


def test_cache_miss_returns_none():
    from src.cache import cache_clear, cache_get, make_cache_key
    cache_clear()
    assert cache_get(make_cache_key("pl_summary", {"year": "9999"})) is None


def test_cache_hit_in_retrieval(monkeypatch):
    """retrieval_sync returns cache_hit=True on second call for same query."""
    from src.cache import cache_clear
    cache_clear()
    from src.agents.retrieval import retrieval_sync
    state = {"intent": "pl_summary", "filters": {"year": "2024"}, "prefetch_results": {}}
    r1 = retrieval_sync(state)
    assert r1["cache_hit"] is False
    r2 = retrieval_sync(state)
    assert r2["cache_hit"] is True
    assert r2["raw_results"] == r1["raw_results"]
