"""Mock tests for LLMRegistry.resolve_context_window."""

from unittest.mock import patch

import pytest

from ai_agent.core.exceptions import ConfigError, ProviderNotFoundError
from ai_agent.core.models.llm import LLM, LLMSettings
from ai_agent.core.registries.llm import LLMRegistry


_PROVIDER = "openai"
_MODEL = LLM(provider=_PROVIDER, model="gpt-4o")
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=4096)


class _StubProvider:
    """Minimal stub satisfying ILLMProvider."""

    def complete(self, request: object) -> object:  # type: ignore[override]
        raise NotImplementedError


@pytest.fixture
def registry() -> LLMRegistry:
    reg = LLMRegistry()
    reg.register(_MODEL, _SETTINGS, _StubProvider())  # type: ignore[arg-type]
    return reg


def test_resolve_context_window_returns_max_input_tokens(registry: LLMRegistry) -> None:
    with patch(
        "ai_agent.core.registries.llm.litellm.get_model_info",
        return_value={"max_input_tokens": 128_000},
    ):
        result = registry.resolve_context_window(_MODEL)

    assert result == 128_000


def test_resolve_context_window_unknown_model_raises_config_error(registry: LLMRegistry) -> None:
    with patch(
        "ai_agent.core.registries.llm.litellm.get_model_info",
        side_effect=Exception("model not mapped"),
    ):
        with pytest.raises(ConfigError):
            registry.resolve_context_window(_MODEL)


def test_resolve_context_window_missing_max_input_tokens_raises_config_error(
    registry: LLMRegistry,
) -> None:
    with patch(
        "ai_agent.core.registries.llm.litellm.get_model_info",
        return_value={"max_input_tokens": None},
    ):
        with pytest.raises(ConfigError):
            registry.resolve_context_window(_MODEL)


def test_resolve_context_window_unregistered_llm_raises_provider_not_found(
    registry: LLMRegistry,
) -> None:
    unknown = LLM(provider="openai", model="gpt-unknown")
    with pytest.raises(ProviderNotFoundError):
        registry.resolve_context_window(unknown)
