"""Exception hierarchy for the AI agent platform."""


class AgentError(Exception):
    """Base class for all AI agent platform exceptions."""


class ConfigError(AgentError):
    """Raised when configuration loading or validation fails."""


class ReasoningError(AgentError):
    """Raised when a strategy-level reasoning failure occurs."""


class LoopDetectedError(ReasoningError):
    """Raised when reasoning exceeds the turn limit."""


class ContextBudgetError(ReasoningError):
    """Raised when the session history is too large to fit in the context window."""


class ToolError(AgentError):
    """Base class for tool-related failures."""


class ToolSchemaError(ToolError):
    """Raised when a tool's schema is invalid at registration time."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not present in the registry."""


class ToolExecutionError(ToolError):
    """Raised when a tool raises an exception during invocation."""


class AgentNotFoundError(AgentError):
    """Raised when a requested agent is not present in the registry."""


class ProviderError(AgentError):
    """Base class for external provider failures."""


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider or model is not registered."""


class CompletionError(ProviderError):
    """Raised when an LLM completion request fails."""


class AuthenticationError(ProviderError):
    """Raised when a provider rejects credentials."""


class RateLimitError(ProviderError):
    """Raised when a provider throttles the request."""


class StateError(AgentError):
    """Raised when an illegal AgentState transition is attempted."""


__all__ = [
    "AgentError",
    "AgentNotFoundError",
    "AuthenticationError",
    "CompletionError",
    "ConfigError",
    "ContextBudgetError",
    "LoopDetectedError",
    "ProviderError",
    "ProviderNotFoundError",
    "RateLimitError",
    "ReasoningError",
    "StateError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolSchemaError",
]
