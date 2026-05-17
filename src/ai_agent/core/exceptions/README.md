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
  - `ToolExecutionError` — tool raised an exception during invocation.
  - `AgentNotFoundError` — requested agent not present in the registry.
  - `ProviderError` — base class for external provider failures.
  - `ProviderNotFoundError` — requested provider or model is not registered.
  - `CompletionError` — LLM completion request failed.
  - `AuthenticationError` — provider rejected credentials.
  - `RateLimitError` — provider throttled the request.
  - `StateError` — illegal agent state transition.
