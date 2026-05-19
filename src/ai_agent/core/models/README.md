# core/models

Shared Pydantic data models used across all layers. All models are frozen (immutable).

## Files

- `__init__.py` — re-exports all domain models.
- `config.py` — startup configuration models (all frozen, validated at load time):
  - `LLMConfig` — model name and sampling defaults.
  - `LLMProviderConfig` — provider name and its list of `LLMConfig` models.
  - `LLMRegistryConfig` — registry of `LLMProviderConfig` entries.
  - `ToolRegistryConfig` — tool configurations exposed to the LLM.
  - `CompactionConfig` — `max_tokens` for compaction summary generation.
  - `LoggingConfig` — minimum log level.
  - `AgentRegistryConfig` — ordered list of `AgentConfig` entries.
  - `ConversationConfig` — global configuration shared across all agents.
- `agent.py` — agent identity and execution state types: `Agent` (identity with `name`), `AgentStatus`, `AgentState`, `StepResult`.
- `strategy.py` — strategy identity and configuration: `Strategy` (type identifier), `StrategyConfig` (base config with `max_turns`; subclass for concrete strategy types).
- `llm.py` — LLM request/response types: `FinishReason`, `LLM`, `LLMUsage` (with `total_tokens` property), `LLMSettings`, `LLMRequest`, `LLMResponse` (with `tool_calls` property), `CompactionResult`.
- `message.py` — conversation message types: `Role`, `Message`.
- `run.py` — run-level types: `RunSettings` (`agent`, `llm`, `settings`, `tools`), `RunResult` (`output`, `turns`, `billed_usage` accumulated across all turns, `context_usage` from the final turn).
- `tool.py` — tool invocation types: `Tool`, `ToolConfig`, `ToolSchema`, `ToolDefinition`, `ToolContext`, `ToolCall`, `ToolResponse`, `ToolResult`.
