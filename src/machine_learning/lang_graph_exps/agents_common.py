"""Shared building blocks for the LangGraph orchestration examples.

Provider/model setup, tools, and the subagent factory live here so that each
`pattern_*.py` file can focus purely on graph structure. Point OPENROUTER_MODEL
at anything on openrouter.ai to swap every agent at once.
"""

from __future__ import annotations

import ast
import operator as op
import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """Single model factory: OpenRouter is OpenAI-compatible, so this is all it takes."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY. Create one at https://openrouter.ai/keys "
            "and add `OPENROUTER_API_KEY=sk-or-...` to your .env"
        )
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        base_url=OPENROUTER_BASE_URL,
        api_key=SecretStr(api_key),
        temperature=temperature,
        default_headers={"X-Title": "langgraph-agents"},
    )


# ---------------------------------------------------------------------------
# Tools the subagents use (deliberately not RAG)
# ---------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
}


def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported arithmetic expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate basic arithmetic, e.g. '12 * 7 + 3'."""
    try:
        return str(_eval(ast.parse(expression, mode="eval")))
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


_FACTS = {
    "langgraph": "LangGraph is a library for building stateful, multi-actor agent applications as graphs.",
    "langchain": "LangChain is a framework for composing LLM applications.",
    "ollama": "Ollama runs large language models locally on your machine.",
    "python": "Python is a programming language created by Guido van Rossum in 1991.",
}


@tool
def lookup_fact(topic: str) -> str:
    """Look up a short stored fact about a topic (try 'langgraph' or 'ollama')."""
    for key, value in _FACTS.items():
        if key in topic.lower():
            return value
    return f"No stored fact for '{topic}'."


# ---------------------------------------------------------------------------
# Subagent factory: each worker is its own compiled ReAct graph
# ---------------------------------------------------------------------------
def build_subagent(name: str, system_prompt: str, tools: list, llm=None):
    """Compile a ReAct subgraph: model <-> tools until it can answer.

    Pass a fake `llm` to build the graph without contacting a real model.
    """
    llm = llm or get_llm()
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    def call_model(state: MessagesState) -> dict:
        prompt = [{"role": "system", "content": system_prompt}, *state["messages"]]
        return {"messages": [llm_with_tools.invoke(prompt)]}

    graph = StateGraph(MessagesState)
    graph.add_node(f"{name}_model", call_model)
    graph.add_edge(START, f"{name}_model")

    if tools:
        graph.add_node(f"{name}_tools", ToolNode(tools))
        graph.add_conditional_edges(f"{name}_model", tools_condition,
                                    {"tools": f"{name}_tools", END: END})
        graph.add_edge(f"{name}_tools", f"{name}_model")
    else:
        graph.add_edge(f"{name}_model", END)

    return graph.compile()


def build_workers(llm=None) -> dict:
    """Build the three specialist subagents.

    Pass a fake `llm` in tests to exercise the graphs without any API calls.
    """
    return {
        "math_agent": build_subagent(
            "math",
            "You are a math specialist. Use the calculator tool for every arithmetic "
            "step, then state the result plainly in one short sentence.",
            [calculator],
            llm,
        ),
        "research_agent": build_subagent(
            "research",
            "You are a research specialist. ALWAYS call the lookup_fact tool before "
            "answering, then quote the returned fact verbatim in one sentence.",
            [lookup_fact],
            llm,
        ),
        "writer_agent": build_subagent(
            "writer",
            "You are a writer. Write the final answer using ONLY the findings already "
            "present in the conversation (numbers computed by math_agent and facts "
            "looked up by research_agent). Address every part of the user's request. "
            "If a finding is missing, say so rather than inventing it.",
            [],
            llm,
        ),
    }


def run_subagent(subgraph, messages: list):
    """Invoke a compiled subagent in isolation; return ONLY its final message.

    Keeping the subagent's private ReAct scratchpad (tool calls, tool results)
    out of the shared conversation is what stops dangling tool calls from
    confusing the other agents.
    """
    payload: MessagesState = {"messages": messages}
    out = subgraph.invoke(payload)
    return out["messages"][-1]
