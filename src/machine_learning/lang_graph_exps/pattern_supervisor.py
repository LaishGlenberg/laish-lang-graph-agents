"""Pattern A: Supervisor orchestration (centralized, sequential handoffs).

A central supervisor LLM decides which subagent acts next. Routing is expressed
with CONDITIONAL EDGES, and every worker reports back to the supervisor:

    START -> supervisor ─┬-> math_agent ────┐
                         ├-> research_agent ┼-> supervisor -> ... -> END
                         └-> writer_agent ──┘

Contrast with pattern_handoffs.py, where the routing lives inside each node's
return value instead of a central router.

Run:
    python src/machine_learning/lang_graph_exps/pattern_supervisor.py
"""

from __future__ import annotations

import operator as op
from typing import Annotated, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agents_common import build_workers, get_llm, run_subagent


class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    # Audit trail of routing decisions; also the supervisor's termination guard.
    history: Annotated[list, op.add]


class SupervisorDecision(BaseModel):
    next: Literal["math_agent", "research_agent", "writer_agent", "FINISH"] = Field(
        description="The worker that should act next, or FINISH when fully answered."
    )


_SUPERVISOR_PROMPT = (
    "You are a supervisor coordinating three workers: math_agent (arithmetic), "
    "research_agent (stored facts), and writer_agent (final write-up). "
    "Given the conversation so far, choose who acts next. Send arithmetic to "
    "math_agent and fact lookups to research_agent. Once the findings are in the "
    "conversation, route to writer_agent to produce the final answer, then reply "
    "FINISH. Never choose FINISH before writer_agent has written the answer."
)


def make_supervisor_node(router, workers: dict):
    """Build the supervisor node around an injected `router` (fake in tests)."""

    def supervisor(state: TeamState) -> dict:
        history = state.get("history", [])

        # Deterministic safety rail: the LLM cannot loop forever. Once the writer
        # has produced an answer, the team is done regardless of what it says next.
        if history and history[-1] == "writer_agent":
            print("  supervisor -> FINISH (writer has produced the answer)")
            return {"next": "FINISH"}

        remaining = [name for name in workers if name not in history]
        options = remaining or ["FINISH"]
        prompt = (
            f"{_SUPERVISOR_PROMPT}\n\nWorkers who have not acted yet: {remaining}. "
            f"Choose exactly one of {options + ['FINISH']}."
        )
        decision = router.invoke(
            [{"role": "system", "content": prompt}, *state["messages"]]
        )

        # Second rail: reject an invalid or repeated choice and fall back safely.
        choice = decision.next
        if choice != "FINISH" and choice not in remaining:
            choice = remaining[0] if remaining else "FINISH"

        print(f"  supervisor -> {choice}")
        return {"next": choice, "history": [choice] if choice != "FINISH" else []}

    return supervisor


def route_supervisor(state: TeamState) -> str:
    return state["next"]


def make_worker_node(name: str, subgraph):
    """Adapter that runs a compiled subagent in isolation and reports one message."""

    def run(state: TeamState) -> dict:
        answer = run_subagent(subgraph, state["messages"])
        print(f"  {name} -> {answer.content[:160]}")
        return {"messages": [answer]}

    return run


def build_graph(workers: dict | None = None, llm=None):
    """Compile the supervisor graph. Inject `workers`/`llm` to test offline."""
    workers = workers or build_workers(llm)
    router = (llm or get_llm()).with_structured_output(SupervisorDecision)

    builder = StateGraph(TeamState)
    builder.add_node("supervisor", make_supervisor_node(router, workers))
    for name, subgraph in workers.items():
        builder.add_node(name, make_worker_node(name, subgraph))
        builder.add_edge(name, "supervisor")

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {**{name: name for name in workers}, "FINISH": END},
    )
    return builder.compile(checkpointer=InMemorySaver())


def demo() -> None:
    print("=" * 70)
    print("PATTERN A: SUPERVISOR (centralized conditional-edge routing)")
    print("=" * 70)
    graph = build_graph()
    task = "How much is 144 / 12, and what is LangGraph?"

    print(f"Task: {task}\n")
    config = {"configurable": {"thread_id": "team-1"}, "recursion_limit": 25}
    for step in graph.stream(
        {"messages": [("user", task)], "next": "", "history": []}, config
    ):
        for node in step:
            print(f"  ran: {node}")

    final = graph.get_state(config).values["messages"][-1]
    print(f"\nFinal answer:\n{final.content}")


if __name__ == "__main__":
    demo()
