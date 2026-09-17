"""Offline test doubles for the LangGraph orchestration patterns.

These let us verify graph *infrastructure* -- wiring, routing, termination,
reducers, state isolation -- with zero API calls, zero cost, and deterministic
results. Only switch to a real model when testing actual agent *behaviour*.

Key idea: the graphs only ever call `.invoke()` on a router/worker and
`.with_structured_output()` / `.bind_tools()` on a model. If we stand in for
those few methods, the entire graph runs offline.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage


class FakeRouter:
    """Drop-in for the object returned by `llm.with_structured_output(...)`.

    Consumes decisions from a queue **shared** with the FakeLLM, so scripted
    routing is used in execution order no matter how many routers a graph
    creates (e.g. one per agent node in the handoff pattern).
    """

    def __init__(self, decisions: list):
        self._decisions = decisions  # shared queue, not a copy
        self.inputs: list = []

    def invoke(self, messages, **kwargs):
        self.inputs.append(messages)
        if not self._decisions:
            raise AssertionError(
                "FakeRouter ran out of scripted decisions -- "
                "add more to FakeLLM(decisions=[...])"
            )
        return self._decisions.pop(0)


class FakeLLM:
    """Minimal chat-model stand-in.

    * `with_structured_output(schema)` -> FakeRouter over a shared decision queue
    * `invoke(...)` -> a scripted AIMessage (used by synthesizer nodes)
    * `bind_tools(...)` -> self (so ReAct subgraphs build without a real model)
    """

    def __init__(self, decisions=None, replies=None):
        self.decisions = list(decisions or [])
        self.replies = list(replies or [])

    def with_structured_output(self, schema, **kwargs):
        return FakeRouter(self.decisions)

    def invoke(self, messages, **kwargs):
        text = self.replies.pop(0) if self.replies else "fake synthesis"
        return AIMessage(content=text)

    def bind_tools(self, tools, **kwargs):
        return self


class FakeWorker:
    """Stands in for a compiled subagent: records calls, returns one message."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list = []

    def invoke(self, input, config=None, **kwargs):
        self.calls.append(input)
        return {"messages": [AIMessage(content=f"{self.name} result")]}


WORKER_NAMES = ("math_agent", "research_agent", "writer_agent")


def make_workers() -> dict[str, FakeWorker]:
    """A fresh dict of fake workers, matching the real worker names."""
    return {name: FakeWorker(name) for name in WORKER_NAMES}
