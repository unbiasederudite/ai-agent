# adapters

Concrete implementations of the provider interfaces defined in `core/`. One sub-package per concern.

## Sub-packages

| Package | Description |
|---------|-------------|
| `llm/` | `ILLMProvider` implementations, one module per LLM provider |
| `runtime/` | Agentic runtime adapters (LangGraph, AutoGen, CrewAI, etc.) |

## Files

- `__init__.py` — package marker, no exports.
