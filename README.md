# machine-learning

Experiments with RAG and LangGraph.

## Layout

- `src/machine_learning/test_rag.py` — minimal RAG pipeline (HuggingFace
  embeddings + Chroma + Ollama).
- `src/machine_learning/lang_graph_exps/test_langgraph.py` — an agentic-RAG
  chatbot built with LangGraph (despite the filename, it is a real
  implementation, not a unit test).
- `src/machine_learning/lang_graph_exps/agents_common.py` — shared model factory,
  tools, and subagent builder used by the orchestration patterns.
- `src/machine_learning/lang_graph_exps/pattern_supervisor.py` — Pattern A:
  centralized supervisor with conditional-edge routing.
- `src/machine_learning/lang_graph_exps/pattern_map_reduce.py` — Pattern B:
  parallel subagents via `Send` + reducer + synthesizer.
- `src/machine_learning/lang_graph_exps/pattern_handoffs.py` — Pattern C:
  decentralized peer handoffs via `Command(goto=..., update=...)`.

## Agentic RAG (LangGraph)

`test_langgraph.py` builds a ReAct-style agent as a LangGraph `StateGraph`:

```
START -> agent -> (tool_calls?) -> tools -> agent -> ... -> END
```

The `agent` node (Ollama `llama3.2:3b`) is bound to a `search_documents` tool
that queries the Chroma vector store. A prebuilt `tools_condition` router and a
`ToolNode` run the tool and feed results back to the agent. An `InMemorySaver`
checkpointer gives the bot per-`thread_id` conversation memory.

Requires a running Ollama server with the model pulled:

```bash
ollama serve
ollama pull llama3.2:3b
python src/machine_learning/lang_graph_exps/test_langgraph.py
```

## Multi-agent orchestration (LangGraph + OpenRouter)

The `lang_graph_exps/pattern_*.py` files each demonstrate one orchestration
pattern in isolation, sharing `agents_common.py`. In every pattern the workers
(`math_agent`, `research_agent`, `writer_agent`) are full compiled `StateGraph`s
(a ReAct loop with their own tools), not plain functions.

- **`pattern_supervisor.py` (Pattern A)** — a central supervisor LLM routes work
  between subagents using **conditional edges**, looping until a writer produces
  the final answer. Deterministic safety rails (routing history + forced
  termination) stop the model from looping forever.
- **`pattern_map_reduce.py` (Pattern B)** — the **`Send` API** fans sub-tasks out
  to specialists that run **in parallel**; an `operator.add` reducer merges their
  outputs and a synthesizer combines them.
- **`pattern_handoffs.py` (Pattern C)** — **no central router**. Each agent
  returns `Command(goto=..., update=...)`, so routing lives inside the nodes
  (the peer-to-peer "swarm" style). Also shows `Command(goto=[Send(...), ...])`
  for parallel fan-out directly from a node's return value.

All patterns share one model behind the `get_llm()` factory in
`agents_common.py`, which talks to OpenRouter (OpenAI-compatible). Configure it
in `.env`:

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini   # optional; any openrouter.ai model ID
```

Free tool-calling models (e.g. `nvidia/nemotron-3-super-120b-a12b:free`) also
work for experimentation. Each pattern is runnable on its own:

```bash
python src/machine_learning/lang_graph_exps/pattern_supervisor.py
python src/machine_learning/lang_graph_exps/pattern_map_reduce.py
python src/machine_learning/lang_graph_exps/pattern_handoffs.py
```
