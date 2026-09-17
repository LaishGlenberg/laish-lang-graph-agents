"""Offline tests for Pattern C: decentralized handoffs via `Command`.

Each agent node builds its own router, but the FakeLLM shares a single decision
queue, so scripted decisions are consumed in execution order.
"""

from __future__ import annotations

from fakes import FakeLLM
from pattern_handoffs import Handoff, build_graph


def _config(thread: str = "t") -> dict:
    return {"configurable": {"thread_id": thread}, "recursion_limit": 15}


def test_routes_decentralized_chain_and_ends(workers):
    llm = FakeLLM(
        decisions=[
            Handoff(next="math_agent"),      # triage decides who starts
            Handoff(next="research_agent"),  # math hands off
            Handoff(next="writer_agent"),    # research hands off
        ]
    )
    graph = build_graph(workers=workers, llm=llm)

    result = graph.invoke(
        {"messages": [("user", "question")], "history": []}, _config()
    )

    assert result["history"] == ["math_agent", "research_agent", "writer_agent"]
    assert result["messages"][-1].content == "writer_agent result"


def test_writer_last_rail_overrides_the_llm(workers):
    """Even if an agent tries to jump straight to the writer, specialists run."""
    llm = FakeLLM(
        decisions=[
            Handoff(next="math_agent"),
            Handoff(next="writer_agent"),  # math tries to skip research
            Handoff(next="writer_agent"),  # research then hands to writer
        ]
    )
    graph = build_graph(workers=workers, llm=llm)

    result = graph.invoke(
        {"messages": [("user", "question")], "history": []}, _config()
    )

    assert "research_agent" in result["history"]
    assert result["history"][-1] == "writer_agent"
