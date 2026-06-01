"""Unit tests for the AgentError exception hierarchy."""

import pytest

from ai_agent.core.exceptions import (
    AgentError,
    AgentNotFoundError,
    AuthenticationError,
    CompletionError,
    ConfigError,
    ContextBudgetError,
    ContextWindowExceededError,
    LoopDetectedError,
    ProviderError,
    ProviderNotFoundError,
    RateLimitError,
    ReasoningError,
    ToolError,
    ToolNotFoundError,
    UserMessageTooLongError,
)


class TestAgentErrorHierarchy:
    """Tests that every exception is rooted at AgentError."""

    def test_config_error_is_agent_error(self) -> None:
        assert issubclass(ConfigError, AgentError)

    def test_tool_error_is_agent_error(self) -> None:
        assert issubclass(ToolError, AgentError)

    def test_completion_error_is_agent_error(self) -> None:
        assert issubclass(CompletionError, AgentError)

    def test_reasoning_error_is_agent_error(self) -> None:
        assert issubclass(ReasoningError, AgentError)

    def test_loop_detected_error_is_reasoning_error(self) -> None:
        assert issubclass(LoopDetectedError, ReasoningError)

    def test_loop_detected_error_is_agent_error(self) -> None:
        assert issubclass(LoopDetectedError, AgentError)

    def test_agent_error_is_exception(self) -> None:
        assert issubclass(AgentError, Exception)

    def test_tool_not_found_error_is_tool_error(self) -> None:
        assert issubclass(ToolNotFoundError, ToolError)

    def test_context_budget_error_is_reasoning_error(self) -> None:
        assert issubclass(ContextBudgetError, ReasoningError)

    def test_agent_not_found_error_is_agent_error(self) -> None:
        assert issubclass(AgentNotFoundError, AgentError)

    def test_provider_error_is_agent_error(self) -> None:
        assert issubclass(ProviderError, AgentError)

    def test_provider_not_found_error_is_provider_error(self) -> None:
        assert issubclass(ProviderNotFoundError, ProviderError)

    def test_completion_error_is_provider_error(self) -> None:
        assert issubclass(CompletionError, ProviderError)

    def test_authentication_error_is_provider_error(self) -> None:
        assert issubclass(AuthenticationError, ProviderError)

    def test_rate_limit_error_is_provider_error(self) -> None:
        assert issubclass(RateLimitError, ProviderError)

    def test_user_message_too_long_error_is_agent_error(self) -> None:
        assert issubclass(UserMessageTooLongError, AgentError)

    def test_context_window_exceeded_error_is_provider_error(self) -> None:
        assert issubclass(ContextWindowExceededError, ProviderError)


class TestAgentErrorChaining:
    """Tests that exception chaining is preserved."""

    def test_config_error_chains_original_cause(self) -> None:
        original = ValueError("bad value")
        with pytest.raises(ConfigError) as exc_info:
            raise ConfigError("config failed") from original
        assert exc_info.value.__cause__ is original

    def test_tool_error_chains_original_cause(self) -> None:
        original = RuntimeError("timeout")
        with pytest.raises(ToolError) as exc_info:
            raise ToolError("tool failed") from original
        assert exc_info.value.__cause__ is original

    def test_completion_error_chains_original_cause(self) -> None:
        original = TimeoutError("deadline exceeded")
        with pytest.raises(CompletionError) as exc_info:
            raise CompletionError("LLM request timed out") from original
        assert exc_info.value.__cause__ is original


class TestAgentErrorMessage:
    """Tests that error messages are accessible."""

    def test_agent_error_stores_message(self) -> None:
        err = AgentError("something went wrong")
        assert str(err) == "something went wrong"

    def test_loop_detected_error_stores_message(self) -> None:
        err = LoopDetectedError("turn limit reached")
        assert str(err) == "turn limit reached"

    def test_tool_not_found_error_stores_message(self) -> None:
        err = ToolNotFoundError("calculator not registered")
        assert str(err) == "calculator not registered"

    def test_user_message_too_long_error_stores_message(self) -> None:
        err = UserMessageTooLongError("message is 500 characters, limit is 100.")
        assert str(err) == "message is 500 characters, limit is 100."
