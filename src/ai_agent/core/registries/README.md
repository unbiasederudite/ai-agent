# core/registries

Runtime registries for agents, LLMs, and tools.

## Files

- `__init__.py` — re-exports `AgentRegistry`, `LLMRegistry`, and `ToolRegistry`.
- `agent.py` — `AgentRegistry`, maps agent names to `Agent` instances.
- `llm.py` — `LLMRegistry`, maps `LLM` identifiers to provider adapters and sampling defaults.
- `tool.py` — `ToolRegistry`, maps `Tool` identifiers to `ITool` adapters and schemas.
