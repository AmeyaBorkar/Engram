# Integrations

Three ways to plug Engram into an agent loop.

## Option 1: framework-agnostic `EngramAgent`

The simplest integration: takes a `Memory` and `ChatProvider`, exposes `chat(message)`.

```python
from engram import Memory, SqliteStorage
from engram.providers.openai import OpenAIChat, OpenAIEmbedder
from engram.integrations import EngramAgent

memory = Memory(
    storage=SqliteStorage("engram.db"),
    embedder=OpenAIEmbedder(),
)
chat = OpenAIChat()
agent = EngramAgent(memory, chat)

turn = agent.chat("What language does the user prefer?")
print(turn.reply)
print(turn.retrieved_context)  # what was injected as system context
print(turn.observed_event_ids)  # ids of the new events recorded this turn
```

Configurable:

- `system_prompt` — base system message; retrieved memories land after it
- `retrieve_k` — how many memories per turn
- `auto_observe=True` — record user message + reply as events for future retrieval
- `include_score` / `include_level` — what to surface in the context bullets

Combined with `record_procedure_outcome` after a successful turn:

```python
agent.record_procedure_outcome(
    situation="user asks about coding style",
    action="suggested ruff + mypy strict",
    outcome=Outcome.SUCCESS,
)
```

## Option 2: LangGraph nodes

Plug Engram into a `StateGraph`:

```python
from langgraph.graph import StateGraph, START, END
from engram.integrations.langgraph import EngramRetrieveNode, EngramObserveNode

builder = StateGraph(MyState)
builder.add_node("retrieve", EngramRetrieveNode(memory))
builder.add_node("agent", my_agent_fn)
builder.add_node("record", EngramObserveNode(memory))
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "agent")
builder.add_edge("agent", "record")
builder.add_edge("record", END)

graph = builder.compile()
result = graph.invoke({"query": "user question"})
```

`EngramRetrieveNode` reads `state["query"]` and writes `state["engram_context"]`.
`EngramObserveNode` reads `state["query"]` and `state["reply"]` and observes both.

Install: `pip install engram-memory[langgraph]`.

## Option 3: LlamaIndex BaseMemory adapter

```python
from llama_index.core.chat_engine import SimpleChatEngine
from engram.integrations.llamaindex import EngramLlamaIndexMemory

llama_memory = EngramLlamaIndexMemory(memory, k=5)
chat_engine = SimpleChatEngine.from_defaults(memory=llama_memory)
```

The adapter duck-types LlamaIndex's `BaseMemory` shape:

- `put(message)` — observes the message content
- `get(input=...)` — returns one system-role message with the formatted memory context
- `get_all()` — returns `[]` (Engram doesn't store messages as turns; it stores events)
- `reset()` — raises `NotImplementedError` (manage the underlying storage backend)

Install: `pip install engram-memory[llamaindex]`.

## Option 4: roll your own with `format_context`

For frameworks Engram doesn't directly support, the smallest integration point is `format_context`:

```python
from engram.integrations import format_context

results = memory.retrieve("question", k=5)
context = format_context(results)
prompt = f"Relevant memories:\n{context}\n\nUser question: ..."
```

Drop-in for any chat loop.

## Async

All three options work with the async surface (`aobserve`, `aretrieve`, `aconsolidate`, etc.). The agent wrapper has sync entry points only today; an async `EngramAgent` lands in v0.4.0 alongside the Postgres backend.
