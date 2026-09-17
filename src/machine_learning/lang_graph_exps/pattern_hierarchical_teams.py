"""Pattern D: Hierarchical teams (nested subgraphs + `Command.PARENT`).

The final step up from Pattern C: teams are themselves subgraphs, and a node
inside a team can hand control back UP to the parent graph. That is what
`graph=Command.PARENT` does -- without it, `goto` can only name a node in the
same (sub)graph.

    parent:  START -> [specialists team] --escalate--> [writer team] --escalate--> END
    team 1:  START -> math_agent -> research_agent --escalate to parent-->
    team 2:  START -> writer_agent --escalate to parent (END)-->

So the parent "chief" orchestrates teams, and each team orchestrates its own
workers. Teams are reusable: mount the same compiled subgraph under several
parent nodes.

GOTCHA demonstrated below: when a subgraph escapes via `Command.PARENT`, its
intermediate local writes are DISCARDED -- only the Command's `update` reaches
the parent. So the escalating node must carry the team's findings up with it.

Run:
    python src/machine_learning/lang_graph_exps/pattern_hierarchical_teams.py
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import TypedDict

from agents_common import build_workers, run_subagent


class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    history: Annotated[list, merge_unique]


def merge_unique(left: list, right: list) -> list:
    """Reducer that appends only new items (order-preserving).

    `operator.add` would concatenate, so a team that correctly passes its full
    history up on escalation would double-count entries the parent already has.
    `add_messages` already dedupes messages by id; this does the same for a
    plain list of names.
    """
    out = list(left)
    for item in right:
        if item not in out:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Team 1: the specialists (its own graph, mounted as a node in the parent)
# ---------------------------------------------------------------------------
def build_specialists_team(workers: dict):
    def math_node(state: TeamState) -> dict:
        answer = run_subagent(workers["math_agent"], state["messages"])
        print(f"  [specialists] math_agent -> {answer.content[:110]}")
        return {"messages": [answer], "history": ["math_agent"]}

    def research_node(state: TeamState) -> Command:
        answer = run_subagent(workers["research_agent"], state["messages"])
        print(f"  [specialists] research_agent -> {answer.content[:110]}")
        print("  [specialists] ESCALATE to parent -> writer_team")
        # Only this `update` survives the escape, so carry the whole team's
        # findings (state[...] + the new answer) up to the parent.
        return Command(
            graph=Command.PARENT,
            goto="writer_team",
            update={
                "messages": [*state["messages"], answer],
                "history": [*state["history"], "research_agent"],
            },
        )

    builder = StateGraph(TeamState)
    builder.add_node("math_agent", math_node)
    builder.add_node("research_agent", research_node)
    builder.add_edge(START, "math_agent")
    builder.add_edge("math_agent", "research_agent")
    return builder.compile()


# ---------------------------------------------------------------------------
# Team 2: the writers
# ---------------------------------------------------------------------------
def build_writer_team(workers: dict):
    def writer_node(state: TeamState) -> Command:
        answer = run_subagent(workers["writer_agent"], state["messages"])
        print(f"  [writer_team] writer_agent -> {answer.content[:110]}")
        print("  [writer_team] ESCALATE to parent -> END")
        return Command(
            graph=Command.PARENT,
            goto=END,
            update={
                "messages": [*state["messages"], answer],
                "history": [*state["history"], "writer_agent"],
            },
        )

    builder = StateGraph(TeamState)
    builder.add_node("writer_agent", writer_node)
    builder.add_edge(START, "writer_agent")
    return builder.compile()


# ---------------------------------------------------------------------------
# The parent "chief" graph: each team is just one node here
# ---------------------------------------------------------------------------
def build_graph(workers: dict | None = None):
    """Compile the parent graph. Inject `workers` to test offline."""
    workers = workers or build_workers()
    builder = StateGraph(TeamState)
    builder.add_node("specialists", build_specialists_team(workers))
    builder.add_node("writer_team", build_writer_team(workers))
    builder.add_edge(START, "specialists")
    # No edges out of the teams: they route themselves into the parent with
    # Command(graph=Command.PARENT, ...).
    return builder.compile(checkpointer=InMemorySaver())


def demo() -> None:
    print("=" * 70)
    print("PATTERN D: HIERARCHICAL TEAMS (nested subgraphs + Command.PARENT)")
    print("=" * 70)
    graph = build_graph()
    task = "How much is 144 / 12, and what is LangGraph?"

    print(f"Task: {task}\n")
    config: RunnableConfig = {"configurable": {"thread_id": "chief-1"}}
    initial: TeamState = {
        "messages": [HumanMessage(content=task)],
        "history": [],
    }
    graph.invoke(initial, config)

    state = graph.get_state(config).values
    print(f"\nTeam history: {state['history']}")
    print(f"\nFinal answer:\n{state['messages'][-1].content}")


if __name__ == "__main__":
    demo()
