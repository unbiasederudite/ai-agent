# core/protocols

All Protocol interfaces defined by the core. Concrete implementations live in `adapters/`.

## Files

- `__init__.py` — re-exports all protocols: `AgentLogger`, `ILLMProvider`, `IReasoningStrategy`, `ITool`.
- `llm.py` — `ILLMProvider`, the Protocol every LLM provider adapter must satisfy. Defines `complete(request)` and `context_window(model)`.
- `tool.py` — `ITool`, the Protocol every tool must satisfy. Requires `config: ToolConfig` and `execute(name, arguments) -> ToolResponse`.
- `strategy.py` — `IReasoningStrategy`, the Protocol every reasoning strategy must satisfy. Requires `config: StrategyConfig` and defines `step(state, request, provider) -> StepResult`.
- `logger.py` — `AgentLogger`, the structured logging interface used throughout `core/`. Defines `debug`, `info`, `warning`, `error`.
