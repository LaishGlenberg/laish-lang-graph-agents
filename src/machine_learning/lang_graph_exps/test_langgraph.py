"""A real LangGraph implementation: an agentic-RAG chatbot.

This is not a toy unit test -- it is a runnable graph. It wires a local Ollama
LLM to a Chroma vector store through LangGraph's ReAct loop:

        START
          |
          v
      +---------+   tool_calls?    +---------+
      |  agent  | ---------------> |  tools  |
      +---------+                  +---------+
          |                            |
          | no tool calls              | results fed back
          v                            v
         END  <--------------------- (loop to agent)

* The **agent** node asks the LLM what to do. The LLM is bound to a
  `search_documents` tool, so it can either answer directly or request a search.
* The **tools** node actually runs the requested tool against Chroma.
* `tools_condition` is a prebuilt router that sends the flow to `tools` when the
  last message contains tool calls, otherwise to `END`.
* An `InMemorySaver` checkpointer gives the bot memory: passing the same
  `thread_id` continues the same conversation.

Run it:
    python src/machine_learning/lang_graph_exps/test_langgraph.py
"""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

MODEL = "llama3.2:3b"
COLLECTION = "rag_documents"
PERSIST_DIR = "./chroma_db"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a `search_documents` tool that "
    "searches a local knowledge base. Use it whenever a question might be "
    "answered by the knowledge base. If the tool returns nothing useful, say so "
    "instead of making things up."
)


# ---------------------------------------------------------------------------
# 1. Retriever: the agent's window into your Chroma vector store
# ---------------------------------------------------------------------------
def build_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
    )
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # Seed the store the first time so the example always works.
    if store._collection.count() == 0:
        store.add_documents(
            [
                Document(page_content="Linux Mint is an Ubuntu-based Linux distribution."),
                Document(page_content="CUDA is NVIDIA's parallel computing platform for GPUs."),
                Document(page_content="RAG stands for Retrieval-Augmented Generation."),
                Document(page_content="Chroma is an open-source vector database."),
            ]
        )

    return store.as_retriever(search_kwargs={"k": 2})


retriever = build_retriever()


# ---------------------------------------------------------------------------
# 2. The tool the agent can call
# ---------------------------------------------------------------------------
@tool
def search_documents(query: str) -> str:
    """Search the local knowledge base for facts about Linux, CUDA, RAG, Chroma.

    Args:
        query: A natural-language search query.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(f"- {d.page_content}" for d in docs)


# ---------------------------------------------------------------------------
# 3. The LLM, bound to the tool
# ---------------------------------------------------------------------------
llm = ChatOllama(model=MODEL, temperature=0)
tools = [search_documents]
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 4. Nodes
# ---------------------------------------------------------------------------
def agent(state: MessagesState) -> dict:
    """Ask the LLM what to do next: call a tool, or answer."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# 5. Assemble the graph
# ---------------------------------------------------------------------------
builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
# Route to "tools" if the agent requested a tool call, otherwise finish.
builder.add_conditional_edges("agent", tools_condition)
# After tools run, go back to the agent so it can use the results.
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# 6. Convenience wrapper
# ---------------------------------------------------------------------------
def chat(message: str, thread_id: str = "default") -> str:
    """Send a message and return the assistant's reply.

    Reusing a `thread_id` keeps the conversation history (memory).
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [("user", message)]}, config)
    return result["messages"][-1].content


if __name__ == "__main__":
    # A fresh thread so reruns start clean.
    thread = {"configurable": {"thread_id": "demo"}}

    question = "What does RAG stand for, and is CUDA involved?"
    print(f"Q: {question}")

    # stream() shows the graph working step by step: agent -> tools -> agent.
    for step in graph.stream({"messages": [("user", question)]}, thread):
        for node, update in step.items():
            print(f"\n[{node}]")
            for msg in update.get("messages", []):
                if getattr(msg, "tool_calls", None):
                    print("  calls:", [c["name"] for c in msg.tool_calls])
                elif msg.content:
                    print(" ", msg.content)

    # A follow-up on the *same* thread proves the checkpointer remembers context.
    follow_up = "And what is Chroma?"
    print(f"\nQ (follow-up): {follow_up}")
    print("A:", chat(follow_up, thread_id="demo"))
