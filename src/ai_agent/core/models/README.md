# core/models

Shared Pydantic data models used across all layers. All models are frozen (immutable).

## Files

- `__init__.py` — re-exports all domain models.
- `budget.py` — `ContextBudget`: tracks context window consumption and signals when compaction is needed.
- `config.py` — startup configuration models (all frozen, validated at load time):
  - `LLMConfig` — model name and sampling defaults.
  - `LLMProviderConfig` — provider name and its available models.
  - `CompactionConfig` — LLM identity, sampling settings, and fill-fraction threshold.
  - `LoggingConfig` — minimum log level.
  - `ConversationConfig` — global configuration: LLM registry, agent registry, tool registry, compaction, logging, and message character limit. Cross-validates all registry references at load time.
- `agent.py` — agent configuration and execution state types: `AgentConfig`, `AgentStatus`, `AgentState`, `StepResult`.
- `strategy.py` — strategy identity and configuration: `StrategyConfig`.
- `llm.py` — LLM request/response types: `LLM`, `LLMSettings`, `LLMRequest`, `LLMResponse`, `LLMUsage`, `CompactionResult`, `FinishReason`.
- `message.py` — conversation message types: `Role`, `Message`.
- `run.py` — run-level types: `RunSettings`, `RunResult`.
- `tool.py` — tool invocation types: `Tool`, `ToolConfig`, `ToolSchema`, `ToolDefinition`, `ToolContext`, `ToolCall`, `ToolResponse`, `ToolResult`.
