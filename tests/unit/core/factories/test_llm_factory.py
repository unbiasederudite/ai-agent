"""Unit tests for LLMFactory."""

import pytest

from ai_agent.core.exceptions import ConfigError
from ai_agent.core.factories.llm import LLMFactory
from ai_agent.core.models.llm import LLMRequest, LLMResponse, LLMUsage, FinishReason
from ai_agent.core.models.message import Message, Role


class _StubProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            message=Message(role=Role.ASSISTANT, content="ok"),
            finish_reason=FinishReason.STOP,
            usage=LLMUsage(input_tokens=1, output_tokens=1),
        )

    def context_window(self, model: str) -> int:
        return 128_000


_PROVIDER_A = _StubProvider()
_PROVIDER_B = _StubProvider()


def _factory(*pairs: tuple[str, _StubProvider]) -> LLMFactory:
    return LLMFactory(implementations=dict(pairs))  # type: ignore[arg-type]


class TestLLMFactoryBuild:
    def test_returns_registered_provider(self) -> None:
        factory = _factory(("openai", _PROVIDER_A))
        assert factory.build("openai") is _PROVIDER_A

    def test_selects_correct_provider_among_multiple(self) -> None:
        factory = _factory(("openai", _PROVIDER_A), ("anthropic", _PROVIDER_B))
        assert factory.build("openai") is _PROVIDER_A
        assert factory.build("anthropic") is _PROVIDER_B

    def test_unknown_provider_raises_config_error(self) -> None:
        factory = _factory()
        with pytest.raises(ConfigError):
            factory.build("openai")

    def test_error_message_contains_unknown_provider(self) -> None:
        factory = _factory()
        with pytest.raises(ConfigError, match="openai"):
            factory.build("openai")

    def test_error_message_lists_available_providers(self) -> None:
        factory = _factory(("anthropic", _PROVIDER_A))
        with pytest.raises(ConfigError, match="anthropic"):
            factory.build("openai")

    def test_same_instance_returned_on_repeated_calls(self) -> None:
        factory = _factory(("openai", _PROVIDER_A))
        assert factory.build("openai") is factory.build("openai")
