"""Shared test fixtures."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# Set dummy env vars before importing anything that reads config
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key")
os.environ.setdefault("LLM_BACKEND", "openai")
os.environ.setdefault("OPENAI_MODEL_REASONING", "gpt-4o-mini")
os.environ.setdefault("OPENAI_MODEL_EXTRACTION", "gpt-4o-mini")
os.environ.setdefault("OPENAI_TEMPERATURE_EXTRACTION", "0.0")
os.environ.setdefault("OPENAI_TEMPERATURE_REASONING", "0.2")
os.environ.setdefault("CORTEX_DATA_PATH", "./cortex.parquet")
os.environ.setdefault("LOG_LEVEL", "INFO")


@pytest.fixture
def mock_llm_response():
    """Factory for mocking OpenAI chat completion responses."""
    def _make(content: str):
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp
    return _make


@pytest.fixture
def mock_chat_complete(monkeypatch):
    """Patch src.llm.local.chat_complete with a controllable AsyncMock."""
    mock = AsyncMock(return_value='{"intent": "pl_summary", "filters": {}}')
    monkeypatch.setattr("src.llm.local.chat_complete", mock)
    return mock


@pytest.fixture
def sample_state():
    """Minimal valid AgentState for tests."""
    return {
        "user_query": "What is the total P&L for 2024?",
        "intent": None,
        "filters": {},
        "raw_results": None,
        "retrieval_error": None,
        "critique": None,
        "judgment": True,
        "needs_clarification": False,
        "clarification_message": None,
        "final_response": None,
        "iterations": 0,
        "error": None,
    }
