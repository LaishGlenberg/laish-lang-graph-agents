"""Offline tests for Pattern D: hierarchical teams via `Command.PARENT`.

These specifically guard the two gotchas discovered while building the pattern:
1. Intermediate subgraph state is discarded on escape, so findings must be
   carried up in the `Command`'s update.
2. The history reducer must deduplicate, or escalation double-counts.
"""

from __future__ import annotations

from pattern_hierarchical_teams import build_graph


def _run(workers):
    graph = build_graph(workers=workers)
    return graph.invoke(
        {"messages": [("user", "question")], "history": []},
        {"configurable": {"thread_id": "t"}},
    )


def test_escalation_preserves_specialist_findings(workers):
    result = _run(workers)

    contents = [m.content for m in result["messages"]]
    assert "math_agent result" in contents
    assert "research_agent result" in contents
    assert result["messages"][-1].content == "writer_agent result"


def test_escalation_history_is_deduplicated(workers):
    result = _run(workers)

    assert result["history"] == ["math_agent", "research_agent", "writer_agent"]
    assert result["history"].count("math_agent") == 1
    assert result["history"].count("research_agent") == 1


def test_all_three_teammates_ran_once(workers):
    _run(workers)

    assert len(workers["math_agent"].calls) == 1
    assert len(workers["research_agent"].calls) == 1
    assert len(workers["writer_agent"].calls) == 1
