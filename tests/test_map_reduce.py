"""Offline tests for Pattern B: map-reduce orchestration.

`Send` fan-out runs workers in parallel, so assertions avoid depending on the
completion order of the parallel branches.
"""

from __future__ import annotations

from fakes import FakeLLM
from pattern_map_reduce import build_graph


def test_fans_out_to_specialists_and_merges_results(workers):
    llm = FakeLLM(replies=["combined answer"])
    graph = build_graph(workers=workers, llm=llm)

    result = graph.invoke({"task": "compute 25*4 and a fact about ollama", "results": []})

    assert len(workers["math_agent"].calls) == 1
    assert len(workers["research_agent"].calls) == 1
    assert len(workers["writer_agent"].calls) == 0  # writer is not in the fan-out
    # The `operator.add` reducer collected both parallel outputs.
    assert len(result["results"]) == 2
    assert result["final"] == "combined answer"


def test_sends_role_specific_instructions(workers):
    llm = FakeLLM(replies=["combined answer"])
    graph = build_graph(workers=workers, llm=llm)

    graph.invoke({"task": "do the thing", "results": []})

    math_instruction = workers["math_agent"].calls[0]["messages"][0][1]
    research_instruction = workers["research_agent"].calls[0]["messages"][0][1]
    assert "calculator" in math_instruction
    assert "lookup_fact" in research_instruction
