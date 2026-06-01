"""Unit tests for ILLMProvider Protocol."""

from ai_agent.core.models import (
    FinishReason,
    LLMRequest,
    LLMSettings,
    LLMResponse,
    LLMUsage,
    Message,
    Role,
)
from ai_agent.core.protocols import ILLMProvider


def _make_settings() -> LLMSettings:
    return LLMSettings(
        temperature=0.7,
        max_tokens=4096,
    )


class _StubLLMProvider:
    """Minimal concrete class satisfying ILLMProvider."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        reply = Message(role=Role.ASSISTANT, content="stub reply")
        return LLMResponse(
            message=reply,
            finish_reason=FinishReason.STOP,
            usage=LLMUsage(input_tokens=1, output_tokens=1),
        )

    def context_window(self, model: str) -> int:
        return 128_000


def _accepts_llm_provider(p: ILLMProvider) -> str:
    return "ok"


class TestILLMProvider:
    """Tests for ILLMProvider Protocol structural subtyping."""

    def test_stub_satisfies_protocol(self) -> None:
        assert _accepts_llm_provider(_StubLLMProvider()) == "ok"

    def test_complete_returns_llm_response(self) -> None:
        provider = _StubLLMProvider()
        msg = Message(role=Role.USER, content="hello")
        request = LLMRequest(model="test_model", settings=_make_settings(), messages=[msg])
        response = provider.complete(request)
        assert isinstance(response, LLMResponse)

    def test_complete_response_role_is_assistant(self) -> None:
        provider = _StubLLMProvider()
        msg = Message(role=Role.USER, content="hi")
        request = LLMRequest(model="test_model", settings=_make_settings(), messages=[msg])
        response = provider.complete(request)
        assert response.message.role == Role.ASSISTANT

    def test_context_window_returns_positive_int(self) -> None:
        provider = _StubLLMProvider()
        assert isinstance(provider.context_window("gpt-4o"), int)
        assert provider.context_window("gpt-4o") >= 1

    def test_class_missing_complete_fails_isinstance(self) -> None:
        """A class without complete() does not satisfy ILLMProvider."""

        class _NoComplete:
            def context_window(self, model: str) -> int:
                return 128_000

        assert not isinstance(_NoComplete(), ILLMProvider)

    def test_class_missing_context_window_fails_isinstance(self) -> None:
        """A class without context_window() does not satisfy ILLMProvider."""

        class _NoContextWindow:
            def complete(self, request: LLMRequest) -> LLMResponse: ...  # type: ignore[return-value]

        assert not isinstance(_NoContextWindow(), ILLMProvider)
