# core/services

Use-case orchestration services.

## Files

- `__init__.py` — exports `Agent`, `CompactionService`, `RunService`, and `ToolService`.
- `agent.py` — `Agent`: the only agent type. Owns system prompt construction and delegates to `RunService`.
- `compaction.py` — `CompactionService`: summarises session history via an LLM to reduce context window usage.
- `run.py` — `RunService`: drives the reasoning loop — calls the strategy each turn until it signals complete or error.
- `tool.py` — `ToolService`: executes tool calls against the registry and returns one result per call.
