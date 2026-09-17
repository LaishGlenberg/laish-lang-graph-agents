"""Tests for the shared building blocks in `agents_common` (offline)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents_common import build_workers, run_subagent
from fakes import FakeLLM


class _FakeSubgraph:
    """A stand-in subagent that returns a scripted message list."""

    def __init__(self, messages):
        self._messages = messages

    def invoke(self, input, config=None, **kwargs):
        return {"messages": self._messages}


def test_run_subagent_returns_only_the_final_message():
    """The private ReAct scratchpad must not leak; only the answer comes back."""
    scratchpad = _FakeSubgraph(
        [
            HumanMessage(content="task"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"}
                ],
            ),
            ToolMessage(content="2", tool_call_id="c1"),
            AIMessage(content="the answer is 2"),
        ]
    )

    result = run_subagent(scratchpad, [("user", "task")])

    assert result.content == "the answer is 2"


def test_build_workers_compiles_all_three_offline():
    llm = FakeLLM(replies=["fake answer"])
    workers = build_workers(llm)

    assert set(workers) == {"math_agent", "research_agent", "writer_agent"}


def test_react_subgraph_runs_with_a_fake_model():
    """A tool-using subagent should still run when the model never calls tools."""
    llm = FakeLLM(replies=["math result"])
    workers = build_workers(llm)

    out = workers["math_agent"].invoke({"messages": [("user", "2+2")]})

    assert out["messages"][-1].content == "math result"
