"""Tests for fuzzy name matching (src/data/fuzzy.py)."""

from __future__ import annotations

from src.data.fuzzy import fuzzy_resolve, resolve_filters


PROPS = {"Building 17", "Building 120", "Building 140"}
TENANTS = {"Tenant 4", "Tenant 6", "Tenant 12"}


def test_fuzzy_resolve_exact():
    assert fuzzy_resolve("Building 17", PROPS) == "Building 17"


def test_fuzzy_resolve_close_match():
    assert fuzzy_resolve("Building 1 7", PROPS) == "Building 17"


def test_fuzzy_resolve_no_match():
    assert fuzzy_resolve("Building 999", PROPS) is None


def test_fuzzy_resolve_empty_candidates():
    assert fuzzy_resolve("Building 17", set()) is None


def test_resolve_filters_string_property(monkeypatch):
    monkeypatch.setattr(
        "src.data.fuzzy.get_cache",
        lambda k: {"available_properties": PROPS, "available_tenants": TENANTS}.get(k, set()),
    )
    result = resolve_filters({"property_name": "Building 17", "year": "2024"})
    assert result["property_name"] == "Building 17"
    assert result["year"] == "2024"


def test_resolve_filters_list_property(monkeypatch):
    monkeypatch.setattr(
        "src.data.fuzzy.get_cache",
        lambda k: {"available_properties": PROPS, "available_tenants": TENANTS}.get(k, set()),
    )
    result = resolve_filters({"property_name": ["Building 17", "Building 120"]})
    assert "Building 17" in result["property_name"]
    assert "Building 120" in result["property_name"]


def test_resolve_filters_fuzzy_property(monkeypatch):
    monkeypatch.setattr(
        "src.data.fuzzy.get_cache",
        lambda k: {"available_properties": PROPS, "available_tenants": TENANTS}.get(k, set()),
    )
    result = resolve_filters({"property_name": "Bldg 17"})
    assert result["property_name"] is not None


def test_resolve_filters_unknown_property_kept(monkeypatch):
    monkeypatch.setattr(
        "src.data.fuzzy.get_cache",
        lambda k: {"available_properties": PROPS, "available_tenants": TENANTS}.get(k, set()),
    )
    result = resolve_filters({"property_name": "Totally Unknown XYZ"})
    assert result["property_name"] == "Totally Unknown XYZ"


def test_resolve_filters_tenant_string(monkeypatch):
    monkeypatch.setattr(
        "src.data.fuzzy.get_cache",
        lambda k: {"available_properties": PROPS, "available_tenants": TENANTS}.get(k, set()),
    )
    result = resolve_filters({"tenant_name": "Tenant 6"})
    assert result["tenant_name"] == "Tenant 6"
