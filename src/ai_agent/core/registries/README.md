# core/registries

Runtime registries for LLMs and tools. Both follow the same pattern:
`register(identity, schema/settings, implementation)` with `setdefault` semantics (first registration wins).

## Files

- `__init__.py` — re-exports `LLMRegistry` and `ToolRegistry`.
- `llm.py` — `LLMRegistry`, maps `LLM` identifiers to provider adapters and sampling defaults. Methods: `register`, `resolve_implementation`, `resolve_settings`, `resolve_models`, and the `llms` property.
- `tool.py` — `ToolRegistry`, maps `Tool` identifiers to ITool adapters and schemas. Methods: `register`, `resolve_implementation`, `resolve_schema`, `resolve_tools`, and the `tools` property.
