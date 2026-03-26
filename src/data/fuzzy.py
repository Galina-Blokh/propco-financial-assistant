"""Fuzzy name matching for property and tenant lookups."""

from __future__ import annotations

from difflib import get_close_matches

from src.data.loader import get_cache


def _normalize_snake(val: str) -> str:
    """Lowercase + replace spaces/hyphens with underscores."""
    return val.lower().replace(" ", "_").replace("-", "_")


def fuzzy_resolve(value: str, candidates: set[str], cutoff: float = 0.80) -> str | None:
    """Return the best fuzzy match from candidates, or None if no match found."""
    if not value or not candidates:
        return None
    if value in candidates:
        return value
    matches = get_close_matches(value, list(candidates), n=1, cutoff=cutoff)
    return matches[0] if matches else None


def resolve_property(name: str) -> str | None:
    """Resolve a property name to a canonical value using fuzzy matching."""
    return fuzzy_resolve(name, get_cache("available_properties") or set())


def resolve_tenant(name: str) -> str | None:
    """Resolve a tenant name to a canonical value using fuzzy matching."""
    return fuzzy_resolve(name, get_cache("available_tenants") or set())


def _resolve_one_or_list(value, resolve_fn) -> str | list[str] | None:
    """Resolve a single name or list of names using the given resolve function."""
    if isinstance(value, list):
        resolved = [resolve_fn(v) or v for v in value if v]
        return resolved if resolved else None
    return resolve_fn(value)


def resolve_ledger_value(val: str, cache_key: str) -> str:
    """Normalize a ledger filter to snake_case, with fuzzy fallback against known values."""
    from difflib import get_close_matches
    normalized = _normalize_snake(val)
    known = get_cache(cache_key) or set()
    if normalized in known:
        return normalized
    matches = get_close_matches(normalized, list(known), n=1, cutoff=0.6)
    return matches[0] if matches else normalized


def resolve_filters(filters: dict) -> dict:
    """Resolve property_name, tenant_name, and ledger fields to canonical values."""
    resolved = dict(filters)
    if resolved.get("property_name"):
        canonical = _resolve_one_or_list(resolved["property_name"], resolve_property)
        if canonical:
            resolved["property_name"] = canonical
    if resolved.get("tenant_name"):
        canonical = _resolve_one_or_list(resolved["tenant_name"], resolve_tenant)
        if canonical:
            resolved["tenant_name"] = canonical
    if resolved.get("ledger_category"):
        resolved["ledger_category"] = resolve_ledger_value(
            resolved["ledger_category"], "available_ledger_categories"
        )
    if resolved.get("ledger_group"):
        resolved["ledger_group"] = resolve_ledger_value(
            resolved["ledger_group"], "available_ledger_groups"
        )
    return resolved
