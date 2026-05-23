# core/protocols

All Protocol interfaces defined by the core.

## Files

- `__init__.py` — re-exports all protocols: `IAgent`, `ILLMProvider`, `IReasoningStrategy`, `ITool`.
- `agent.py` — `IAgent`, the Protocol every agent implementation must satisfy.
- `llm.py` — `ILLMProvider`, the Protocol every LLM provider adapter must satisfy.
- `tool.py` — `ITool`, the Protocol every tool must satisfy.
- `strategy.py` — `IReasoningStrategy`, the Protocol every reasoning strategy must satisfy.
