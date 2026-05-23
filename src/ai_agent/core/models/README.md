# core/models

Shared Pydantic data models used across all layers. All models are frozen (immutable).

## Files

- `__init__.py` — re-exports all domain models.
- `budget.py` — `ContextBudget`: tracks context window consumption; `token_limit`, `should_compact`, `utilization` properties; `update()`, `reset_usage()`, `recalibrate()` mutation methods.
- `config.py` — startup configuration models (all frozen, validated at load time):
  - `LLMConfig` — model name and sampling defaults.
  - `LLMProviderConfig` — provider name and its list of `LLMConfig` models.
  - `LLMRegistryConfig` — registry of `LLMProviderConfig` entries.
  - `ToolRegistryConfig` — tool configurations exposed to the LLM.
  - `CompactionConfig` — `llm` (validated against registry), `temperature`, `threshold` (0–1 fill fraction), `max_tokens`.
  - `LoggingConfig` — minimum log level.
  - `AgentRegistryConfig` — ordered list of `AgentConfig` entries.
  - `ConversationConfig` — global configuration: `llm_registry`, `agent_registry`, `default_agent` (active agent at start, validated against registry), `tool_registry`, `compaction` (compaction.llm validated against registry), `logging`, `message_char_limit` (max characters per user message).
- `agent.py` — agent identity, configuration, and execution state types: `Agent` (identity with `type` and `name`), `AgentConfig(Agent)` (empty base; subclass and add fields per implementation), `AgentStatus`, `AgentState`, `StepResult`.
- `strategy.py` — strategy identity and configuration: `Strategy` (type identifier), `StrategyConfig` (base config with `max_turns`; subclass for concrete strategy types).
- `llm.py` — LLM request/response types: `FinishReason`, `LLM`, `LLMUsage` (with `total_tokens` property), `LLMSettings`, `LLMRequest`, `LLMResponse` (with `tool_calls` property), `CompactionResult`.
- `message.py` — conversation message types: `Role`, `Message`.
- `run.py` — run-level types: `RunSettings` (`agent`, `llm`, `settings`, `tools`), `RunResult` (`output`, `turns`, `billed_usage` accumulated across all turns, `context_usage` from the final turn).
- `tool.py` — tool invocation types: `Tool`, `ToolConfig`, `ToolSchema`, `ToolDefinition`, `ToolContext`, `ToolCall`, `ToolResponse`, `ToolResult`.
