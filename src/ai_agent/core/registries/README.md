# core/registries

Runtime registries for agents, LLMs, and tools. All follow the same pattern:
`register(identity, ...)` with `setdefault` semantics (first registration wins).

## Files

- `__init__.py` — re-exports `AgentRegistry`, `LLMRegistry`, and `ToolRegistry`.
- `agent.py` — `AgentRegistry`, maps agent names to `AgentNode` instances. Methods: `register(agent, node)`, `resolve(agent) → AgentNode`, and the `agents` property (list of registered names).
- `llm.py` — `LLMRegistry`, maps `LLM` identifiers to provider adapters and sampling defaults. Methods: `register`, `resolve_implementation`, `resolve_settings`, `resolve_models`, and the `llms` property.
- `tool.py` — `ToolRegistry`, maps `Tool` identifiers to ITool adapters and schemas. Methods: `register`, `resolve_implementation`, `resolve_definition` (returns a `ToolDefinition` for a given `Tool`), `resolve_tools`, and the `tools` property.
