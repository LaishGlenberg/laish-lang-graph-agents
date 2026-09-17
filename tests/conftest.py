"""Shared pytest fixtures for the LangGraph orchestration tests.

The `workers` fixture gives each test its own set of fake subagents so tests can
assert on call order/count without any network access.
"""

from __future__ import annotations

import pytest

from fakes import FakeLLM, make_workers


@pytest.fixture
def workers():
    return make_workers()


@pytest.fixture
def fake_llm():
    """Factory fixture: `fake_llm(decisions=[...], replies=[...])`."""
    return FakeLLM
