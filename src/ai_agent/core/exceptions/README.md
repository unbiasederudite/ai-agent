# core/exceptions

Platform exception hierarchy. All exceptions inherit from `AgentError`.

## Files

- `__init__.py` — defines and exports the full exception hierarchy:
  - `AgentError` — base class for all platform exceptions.
  - `ConfigError` — configuration loading or validation failure.
  - `ReasoningError` — strategy-level reasoning failure.
  - `LoopDetectedError` — reasoning exceeded the turn limit.
  - `ContextBudgetError` — session history too large for the context window.
  - `ToolError` — base class for tool-related failures.
  - `ToolSchemaError` — invalid tool schema at registration time.
  - `ToolNotFoundError` — requested tool not present in the registry.
  - `AgentNotFoundError` — requested agent not present in the registry.
  - `UserMessageTooLongError` — user message exceeds the configured character limit.
  - `ProviderError` — base class for external provider failures.
  - `ProviderNotFoundError` — requested provider or model is not registered.
  - `ContextWindowExceededError` — message list exceeds the provider's context window.
  - `CompletionError` — LLM completion request failed.
  - `AuthenticationError` — provider rejected credentials.
  - `RateLimitError` — provider throttled the request.
