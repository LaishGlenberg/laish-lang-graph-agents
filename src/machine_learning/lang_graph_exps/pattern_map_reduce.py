"""Pattern B: Map-reduce orchestration (parallel subagents).

A fan-out step dispatches sub-tasks to specialists that run CONCURRENTLY, a
reducer merges their outputs, and a synthesizer combines everything:

    START -> fan_out ─┬-> worker(math_agent) ────┐
                      └-> worker(research_agent) ┼-> synthesizer -> END

The `Send` API is what makes this parallel: fan_out returns one Send per
sub-task, and each Send lands on the same "worker" node with a different
payload. `Annotated[list, operator.add]` is what makes the concurrent results
merge instead of clobbering each other.

Contrast with pattern_supervisor.py (sequential) and pattern_handoffs.py
(decentralized routing).

Run:
    python src/machine_learning/lang_graph_exps/pattern_map_reduce.py
"""

from __future__ import annotations

import operator as op
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from agents_common import WORKERS, get_llm, run_subagent


class MapReduceState(TypedDict):
    task: str
    results: Annotated[list, op.add]  # reducer collects parallel outputs
    final: str


def fan_out(state: MapReduceState):
    """Split the request and dispatch one sub-task per specialist concurrently."""
    subtasks = {
        "math_agent": f"Compute the arithmetic here with the calculator: {state['task']}",
        "research_agent": f"Look up the factual part here with lookup_fact: {state['task']}",
    }
    return [
        Send("worker", {"role": role, "instruction": instruction})
        for role, instruction in subtasks.items()
    ]


def worker(state: dict) -> dict:
    """Runs *inside* the fan-out; invokes the matching compiled subagent."""
    role = state["role"]
    answer = run_subagent(WORKERS[role], [("user", state["instruction"])])
    return {"results": [f"### {role}\n{answer.content}"]}


def synthesizer(state: MapReduceState) -> dict:
    joined = "\n\n".join(state["results"])
    prompt = f"Combine these worker findings into one concise final answer.\n\n{joined}"
    return {"final": get_llm().invoke(prompt).content}


def build_graph():
    builder = StateGraph(MapReduceState)
    builder.add_node("worker", worker)
    builder.add_node("synthesizer", synthesizer)
    # A conditional edge from START may return Send objects to fan out.
    builder.add_conditional_edges(START, fan_out, ["worker"])
    builder.add_edge("worker", "synthesizer")
    builder.add_edge("synthesizer", END)
    return builder.compile()


def demo() -> None:
    print("=" * 70)
    print("PATTERN B: MAP-REDUCE (parallel subagents via Send)")
    print("=" * 70)
    graph = build_graph()
    task = "Give me a fact about Ollama and compute 25 * 4."

    print(f"Task: {task}\n")
    result = graph.invoke({"task": task, "results": []})
    print("Parallel worker outputs:")
    for chunk in result["results"]:
        print("  " + chunk.replace("\n", "\n  "))
    print(f"\nSynthesised answer:\n{result['final']}")


if __name__ == "__main__":
    demo()
