# core/protocols

All Protocol interfaces defined by the core.

## Files

- `__init__.py` — re-exports all protocols: `ILLMProvider`, `IReasoningStrategy`, `ITool`.
- `llm.py` — `ILLMProvider`, the Protocol every LLM provider adapter must satisfy.
- `tool.py` — `ITool`, the Protocol every tool must satisfy.
- `strategy.py` — `IReasoningStrategy`, the Protocol every reasoning strategy must satisfy.
