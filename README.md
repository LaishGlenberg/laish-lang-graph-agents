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
- `src/machine_learning/lang_graph_exps/pattern_hierarchical_teams.py` — Pattern
  D: nested teams that escalate to the parent via `Command(graph=Command.PARENT)`.

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

Each `lang_graph_exps/pattern_*.py` file isolates **one** orchestration pattern
and runs on its own; they all share `agents_common.py`. Every file exposes a
`build_graph()` and a `demo()`. In every pattern the workers (`math_agent`,
`research_agent`, `writer_agent`) are full compiled `StateGraph`s (a ReAct loop
with their own tools), not plain functions — that is what makes them subagents.

### The progression

| File | Pattern | Routing style | Key LangGraph features |
| --- | --- | --- | --- |
| `pattern_supervisor.py` | A. Supervisor | centralized, conditional edges | `add_conditional_edges`, `with_structured_output`, checkpointer |
| `pattern_map_reduce.py` | B. Map-reduce | fan-out / fan-in | `Send`, reducers |
| `pattern_handoffs.py` | C. Handoffs | decentralized, node returns it | `Command(goto=..., update=...)` |
| `pattern_hierarchical_teams.py` | D. Nested teams | team escalates to parent | `Command(graph=Command.PARENT)` |

### Pattern details

- **A. Supervisor** — a central supervisor LLM routes work between subagents
  using conditional edges, looping until a writer produces the final answer.
  Deterministic safety rails (routing history + forced termination) stop the
  model from looping forever.
- **B. Map-reduce** — the `Send` API fans sub-tasks out to specialists that run
  **in parallel**; an `operator.add` reducer merges their outputs, and a
  synthesizer combines them.
- **C. Handoffs** — **no central router**. Each agent returns
  `Command(goto=..., update=...)`, so routing lives inside the nodes (the
  peer-to-peer "swarm" style). Also demonstrates
  `Command(goto=[Send(...), Send(...)])` for parallel fan-out from a node.
- **D. Hierarchical teams** — teams are subgraphs. A team escalates back up to
  the parent "chief" graph with `Command(graph=Command.PARENT, goto=...)`; the
  parent orchestrates teams while each team orchestrates its own workers. Teams
  are reusable components.

### Lessons baked into the code (the tricky bits)

1. **Subagent isolation** — a worker's private ReAct scratchpad (tool calls and
   tool results) must not leak into the shared conversation. `run_subagent()`
   returns only the final message; sharing raw `messages` leaves dangling tool
   calls that confuse the other models.
2. **Deterministic safety rails** — an LLM router can loop forever. Pattern A
   tracks a routing `history` and forces termination once the writer has run.
3. **`Command.PARENT` discards intermediate state** — when a subgraph escapes to
   its parent, only the `Command`'s `update` reaches the parent. The escalating
   node must carry the team's findings up with it, or they are lost.
4. **Reducers define merge semantics** — `add_messages` dedupes by message id,
   `operator.add` simply concatenates, and Pattern D defines a `merge_unique`
   reducer for names so escalation does not double-count.

### Setup

All patterns share one model behind the `get_llm()` factory in
`agents_common.py`, which talks to OpenRouter (OpenAI-compatible). Configure it
in `.env`:

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini   # optional; any openrouter.ai model ID
```

Free tool-calling models (e.g. `nvidia/nemotron-3-super-120b-a12b:free`) also
work for experimentation.

### Run

```bash
python src/machine_learning/lang_graph_exps/pattern_supervisor.py
python src/machine_learning/lang_graph_exps/pattern_map_reduce.py
python src/machine_learning/lang_graph_exps/pattern_handoffs.py
python src/machine_learning/lang_graph_exps/pattern_hierarchical_teams.py
```

## Testing the graphs offline (no API key, no cost)

Graphs are code, so verify their **infrastructure** -- wiring, routing,
termination, reducers, state isolation -- with fakes *before* spending tokens on
real models. The whole suite runs in a couple of seconds with **no API key and
no network**:

```bash
uv run pytest            # or: .venv/bin/python -m pytest
```

### How it works

Every graph only ever calls a handful of methods on its collaborators:
`router.invoke(...)`, `worker.invoke(...)`, `llm.bind_tools(...)`, and
`llm.with_structured_output(...)`. The test doubles in `tests/fakes.py` stand in
for exactly those, so an entire graph runs offline:

- **`FakeLLM`** -- `with_structured_output()` returns a `FakeRouter` that pops
  scripted decisions from a queue shared across every router a graph creates
  (important for the handoff pattern, where each node has its own router).
  `invoke()` returns a scripted reply for synthesizer nodes.
- **`FakeWorker`** -- records every call and returns a single canned message, so
  tests can assert on call order/count.
- **`tests/conftest.py`** -- the `workers` fixture gives each test a fresh set of
  fakes.

To make this possible, the pattern modules build their model/workers lazily and
accept injected dependencies, e.g. `build_graph(workers=..., llm=...)`.

### What each test file covers

| File | Asserts |
| --- | --- |
| `tests/test_common.py` | `run_subagent` returns only the final message (scratchpad isolation) |
| `tests/test_supervisor.py` | routing order, writer termination rail, repeat-choice fallback, message isolation |
| `tests/test_map_reduce.py` | fan-out to the right specialists, reducer merge, role-specific instructions |
| `tests/test_handoffs.py` | `Command` chain routing, writer-last rail |
| `tests/test_hierarchical_teams.py` | `Command.PARENT` findings survive escape, history dedupe |

### The workflow to follow

1. Change a graph (add a node, a rail, a reducer).
2. Run the offline suite -- it should still pass, and you can script new
   decisions in the relevant test file.
3. Only then run the `pattern_*.py` demo with a real model to check actual agent
   *behaviour* (answer quality, tool use).

## Editor setup (why Pylance flags `import agents_common`)

The pattern modules are loose sibling files that import each other by bare name
(`import agents_common`, and `import fakes` in the tests). This works at runtime
because a directly-run script adds its own folder to `sys.path`, and pytest adds
the same folders via `[tool.pytest.ini_options] pythonpath`. Pylance/Pyright does
not read either of those, so it reports the imports as unresolved.

`pyrightconfig.json` at the repo root fixes it by adding those folders to
`extraPaths`. If VS Code still complains, add the same paths to
`.vscode/settings.json`:

```json
{ "python.analysis.extraPaths": ["src/machine_learning/lang_graph_exps", "tests"] }
```

(An alternative is to make `lang_graph_exps` a real package with an
`__init__.py` and use fully-qualified `machine_learning.lang_graph_exps...`
imports, but that makes running a single file directly less convenient.)

## Static typing (why Pylance/Pyright complains, and the conventions we use)

LangGraph and LangChain validate state at *runtime* (Pydantic coercion, TypedDict
construction, message-tuple shorthand), which Pylance cannot see. The patterns
now follow a few conventions so `pyright` reports **0 errors** on
`src/machine_learning/lang_graph_exps`:

- **`api_key=str`** -- `ChatOpenAI` annotates `api_key` as `SecretStr`; Pydantic
  coerces a `str` at runtime but the checker can't. Wrap with
  `SecretStr(api_key)` (`agents_common.get_llm`).
- **`with_structured_output(...)` returns `dict`** -- the schema type is lost, so
  `.invoke(...)` results are wrapped in `cast(SupervisorDecision, ...)` /
  `cast(Handoff, ...)`.
- **Message tuples** -- use `HumanMessage(content=...)` instead of the
  `("user", ...)` shorthand, which isn't statically an `AnyMessage`.
- **Graph inputs / config** -- annotate with the state TypedDict (`initial:
  TeamState = {...}`) and `config: RunnableConfig = {...}` rather than letting
  Pyright infer a bare `dict`.
- **`Send`-payload nodes** -- a node fed by `Send` sees a different schema than
  the graph state, which LangGraph's stubs can't express; those `add_node` calls
  carry a documented `# type: ignore[arg-type]`.

Run the checker with:

```bash
uv run --with pyright pyright src/machine_learning/lang_graph_exps
```

`pyrightconfig.json`'s `"//"` key is a JSON comment; Pyright prints a harmless
"unrecognized setting" line for it.
