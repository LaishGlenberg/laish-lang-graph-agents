"""Offline tests for Pattern A: supervisor orchestration.

We script the router's decisions and use fake workers, so these assert on the
graph's *infrastructure* -- routing order, termination rails, state isolation --
without any API calls.
"""

from __future__ import annotations

from fakes import FakeLLM
from pattern_supervisor import SupervisorDecision, build_graph


def _config(thread: str = "t") -> dict:
    return {"configurable": {"thread_id": thread}, "recursion_limit": 15}


def _invoke(graph, config=None):
    return graph.invoke(
        {"messages": [("user", "question")], "next": "", "history": []},
        config or _config(),
    )


def test_runs_each_worker_then_finishes(workers):
    llm = FakeLLM(
        decisions=[
            SupervisorDecision(next="math_agent"),
            SupervisorDecision(next="research_agent"),
            SupervisorDecision(next="writer_agent"),
        ]
    )
    graph = build_graph(workers=workers, llm=llm)

    result = _invoke(graph)

    assert len(workers["math_agent"].calls) == 1
    assert len(workers["research_agent"].calls) == 1
    assert len(workers["writer_agent"].calls) == 1
    assert result["history"] == ["math_agent", "research_agent", "writer_agent"]
    # Final message is the writer's isolated output, not a scratchpad.
    assert result["messages"][-1].content == "writer_agent result"


def test_subagent_scratchpad_does_not_leak_into_team_messages(workers):
    llm = FakeLLM(
        decisions=[
            SupervisorDecision(next="math_agent"),
            SupervisorDecision(next="research_agent"),
            SupervisorDecision(next="writer_agent"),
        ]
    )
    graph = build_graph(workers=workers, llm=llm)

    result = _invoke(graph)

    # Exactly: the user message + one message per worker.
    assert len(result["messages"]) == 4


def test_terminates_after_writer_even_if_llm_keeps_routing(workers):
    """The deterministic rail must stop a runaway router."""
    llm = FakeLLM(decisions=[SupervisorDecision(next="writer_agent")] * 5)
    graph = build_graph(workers=workers, llm=llm)

    result = _invoke(graph)

    assert len(workers["writer_agent"].calls) == 1
    assert result["history"] == ["writer_agent"]


def test_falls_back_when_llm_repeats_a_worker(workers):
    """An invalid/repeated choice falls back to the first unrun worker."""
    llm = FakeLLM(
        decisions=[
            SupervisorDecision(next="math_agent"),
            SupervisorDecision(next="math_agent"),  # repeat -> rail kicks in
            SupervisorDecision(next="writer_agent"),
        ]
    )
    graph = build_graph(workers=workers, llm=llm)

    _invoke(graph)

    assert len(workers["research_agent"].calls) == 1
