"""Pattern C: Decentralized handoffs with `Command(goto=..., update=...)`.

There is no central supervisor here. Each agent decides who acts next and
RETURNS that routing decision from the node itself:

    return Command(goto="research_agent", update={"messages": [answer]})

`Command` does two things in a single return value:
  * `update` merges into the graph state (like a normal node return), and
  * `goto` routes to the next node -- a node name, `END`, or a list of `Send`
    objects for a parallel fan-out.

This is the "swarm" / peer-to-peer style: routing logic lives inside the nodes,
so adding a new agent means teaching its peers to hand off to it, rather than
editing a central router (contrast pattern_supervisor.py).

    START -> triage --handoff--> agent --handoff--> agent --handoff--> END

Run:
    python src/machine_learning/lang_graph_exps/pattern_handoffs.py
"""

from __future__ import annotations

import operator as op
from typing import Annotated, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agents_common import WORKERS, get_llm, run_subagent


class SwarmState(TypedDict):
    messages: Annotated[list, add_messages]
    history: Annotated[list, op.add]


class Handoff(BaseModel):
    next: Literal["math_agent", "research_agent", "writer_agent", "DONE"] = Field(
        description="The peer that should act next, or DONE if the request is answered."
    )


_HANDOFF_RULES = (
    "You are {name}, one peer in a team of equals: math_agent (arithmetic), "
    "research_agent (stored facts), writer_agent (final write-up). Decide which "
    "peer should act next, or DONE if the request is fully answered. Never hand "
    "off to yourself or to a peer that has already acted. writer_agent always "
    "writes the final answer last."
)


def _next_peer(decider, name: str, state: SwarmState, answer) -> Command:
    """Ask the LLM for the next peer, apply safety rails, and return a Command."""
    history = state.get("history", [])
    peers = [w for w in WORKERS if w not in history and w != name]
    # Keep the writer for last so the demo reliably exercises the specialists.
    remaining = [w for w in peers if w != "writer_agent"] or peers

    prompt = f"{_HANDOFF_RULES.format(name=name)}\nPeers still able to act: {remaining}."
    decision = decider.invoke(
        [{"role": "system", "content": prompt}, *state["messages"], answer]
    )

    nxt = decision.next
    if nxt not in remaining:  # rail: never repeat a peer or loop forever
        nxt = remaining[0] if remaining else "DONE"

    print(f"  {name} --handoff--> {nxt}")
    return Command(
        goto=END if nxt == "DONE" else nxt,
        update={"messages": [answer], "history": [name]},
    )


def make_agent_node(name: str, subgraph):
    """Wrap a compiled subagent; its routing is returned via `Command`."""
    decider = get_llm().with_structured_output(Handoff)

    def node(state: SwarmState) -> Command:
        answer = run_subagent(subgraph, state["messages"])
        print(f"  {name} -> {answer.content[:140]}")

        # The writer is terminal: it ends the graph itself.
        if name == "writer_agent":
            print(f"  {name} --handoff--> DONE")
            return Command(goto=END, update={"messages": [answer], "history": [name]})

        return _next_peer(decider, name, state, answer)

    return node


def triage(state: SwarmState) -> Command:
    """Entry point: use `Command` to pick the first specialist."""
    decider = get_llm().with_structured_output(Handoff)
    decision = decider.invoke(
        [
            {
                "role": "system",
                "content": "You dispatch work. Who should start: math_agent, "
                "research_agent, or writer_agent? Choose one.",
            },
            *state["messages"],
        ]
    )
    first = decision.next if decision.next in WORKERS else "math_agent"
    print(f"  triage --handoff--> {first}")
    return Command(goto=first)


def build_graph():
    builder = StateGraph(SwarmState)
    builder.add_node("triage", triage)
    for name, subgraph in WORKERS.items():
        builder.add_node(name, make_agent_node(name, subgraph))
    builder.add_edge(START, "triage")
    # No conditional edges: every node routes itself by returning a Command.
    return builder.compile(checkpointer=InMemorySaver())


def demo() -> None:
    print("=" * 70)
    print("PATTERN C: DECENTRALIZED HANDOFFS (Command(goto=..., update=...))")
    print("=" * 70)
    graph = build_graph()
    task = "How much is 144 / 12, and what is LangGraph?"

    print(f"Task: {task}\n")
    config = {"configurable": {"thread_id": "swarm-1"}, "recursion_limit": 25}
    graph.invoke({"messages": [("user", task)], "history": []}, config)

    final = graph.get_state(config).values["messages"][-1]
    print(f"\nFinal answer:\n{final.content}")


# ---------------------------------------------------------------------------
# Bonus: `Command` can fan out too, by passing a list of `Send`.
# ---------------------------------------------------------------------------
class FanoutState(TypedDict):
    results: Annotated[list, op.add]


def dispatch(state: FanoutState) -> Command:
    # goto as a list == parallel branches, all from one node's return value.
    return Command(
        goto=[Send("echo", {"label": "a"}), Send("echo", {"label": "b"})]
    )


def echo(state: dict) -> dict:
    return {"results": [f"echo:{state['label']}"]}


def demo_command_fanout() -> None:
    print("\n" + "=" * 70)
    print("BONUS: Command(goto=[Send(...), Send(...)]) fans out from a node")
    print("=" * 70)
    builder = StateGraph(FanoutState)
    builder.add_node("dispatch", dispatch)
    builder.add_node("echo", echo)
    builder.add_edge(START, "dispatch")
    builder.add_edge("echo", END)
    print(" ", builder.compile().invoke({"results": []}))


if __name__ == "__main__":
    demo()
    demo_command_fanout()
