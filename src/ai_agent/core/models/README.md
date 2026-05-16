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
  - `AgentConfig` — per-agent configuration: `llm`, `tools`, `system_prompt`, `strategy`.
  - `AgentRegistry` — ordered list of `AgentConfig` entries.
  - `ConversationConfig` — global configuration shared across all agents.
- `strategy.py` — strategy identity and configuration: `Strategy` (type identifier), `StrategyConfig` (base config with `max_turns`; subclass for concrete strategy types).
- `llm.py` — LLM request/response types: `FinishReason`, `LLM`, `LLMUsage`, `LLMSettings`, `LLMRequest`, `LLMResponse`.
- `message.py` — conversation message types: `Role`, `Message`.
- `run.py` — run-level types: `RunSettings`, `RunResult`.
- `state.py` — execution state types: `AgentStatus`, `AgentState`, `StepResult`.
- `tool.py` — tool invocation types: `Tool`, `ToolConfig`, `ToolSchema`, `ToolDefinition`, `ToolContext`, `ToolCall`, `ToolResponse`, `ToolResult`.
