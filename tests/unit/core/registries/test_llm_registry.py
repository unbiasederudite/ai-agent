"""Unit tests for LLMRegistry."""

import pytest

from ai_agent.core.exceptions import ProviderNotFoundError
from ai_agent.core.models.llm import (
    LLM,
    LLMSettings,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    FinishReason,
)
from ai_agent.core.models.message import Message, Role
from ai_agent.core.registries.llm import LLMRegistry


_PROVIDER_A = "openai"
_PROVIDER_B = "anthropic"
_MODEL_A1 = LLM(provider=_PROVIDER_A, model="gpt-4o")
_MODEL_A2 = LLM(provider=_PROVIDER_A, model="gpt-4o-mini")
_MODEL_B1 = LLM(provider=_PROVIDER_B, model="claude-3-5-sonnet")
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=4096)


class _StubProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            message=Message(role=Role.ASSISTANT, content="stub"),
            finish_reason=FinishReason.STOP,
            usage=LLMUsage(input_tokens=1, output_tokens=1),
        )

    def context_window(self, model: str) -> int:
        return 128_000


class TestLLMRegistryRegister:
    def test_register_valid_llm(self) -> None:
        registry = LLMRegistry()
        registry.register(_MODEL_A1, _SETTINGS, _StubProvider())

    def test_register_duplicate_silently_ignored(self) -> None:
        registry = LLMRegistry()
        p1 = _StubProvider()
        p2 = _StubProvider()
        registry.register(_MODEL_A1, _SETTINGS, p1)
        registry.register(_MODEL_A1, _SETTINGS, p2)
        assert registry.resolve_implementation(_PROVIDER_A) is p1

    def test_register_multiple_models_same_provider(self) -> None:
        registry = LLMRegistry()
        registry.register(_MODEL_A1, _SETTINGS, _StubProvider())
        registry.register(_MODEL_A2, _SETTINGS, _StubProvider())
        assert len(registry.llms) == 2

    def test_register_different_providers_independent(self) -> None:
        registry = LLMRegistry()
        pa = _StubProvider()
        pb = _StubProvider()
        registry.register(_MODEL_A1, _SETTINGS, pa)
        registry.register(_MODEL_B1, _SETTINGS, pb)
        assert registry.resolve_implementation(_PROVIDER_A) is pa
        assert registry.resolve_implementation(_PROVIDER_B) is pb


class TestLLMRegistryResolveImplementation:
    def test_returns_registered_provider(self) -> None:
        registry = LLMRegistry()
        provider = _StubProvider()
        registry.register(_MODEL_A1, _SETTINGS, provider)
        assert registry.resolve_implementation(_PROVIDER_A) is provider

    def test_unknown_provider_raises_provider_not_found(self) -> None:
        registry = LLMRegistry()
        with pytest.raises(ProviderNotFoundError):
            registry.resolve_implementation("nonexistent")

    def test_shared_across_models_of_same_provider(self) -> None:
        registry = LLMRegistry()
        provider = _StubProvider()
        registry.register(_MODEL_A1, _SETTINGS, provider)
        registry.register(_MODEL_A2, _SETTINGS, _StubProvider())
        assert registry.resolve_implementation(_PROVIDER_A) is provider


class TestLLMRegistryResolveSettings:
    def test_returns_registered_settings(self) -> None:
        registry = LLMRegistry()
        registry.register(_MODEL_A1, _SETTINGS, _StubProvider())
        assert registry.resolve_settings(_MODEL_A1) is _SETTINGS

    def test_unknown_llm_raises_provider_not_found(self) -> None:
        registry = LLMRegistry()
        with pytest.raises(ProviderNotFoundError):
            registry.resolve_settings(_MODEL_A1)

    def test_distinct_settings_per_model(self) -> None:
        registry = LLMRegistry()
        settings_a = LLMSettings(temperature=0.1, max_tokens=1024)
        settings_b = LLMSettings(temperature=0.9, max_tokens=8192)
        registry.register(_MODEL_A1, settings_a, _StubProvider())
        registry.register(_MODEL_A2, settings_b, _StubProvider())
        assert registry.resolve_settings(_MODEL_A1) is settings_a
        assert registry.resolve_settings(_MODEL_A2) is settings_b


class TestLLMRegistryResolveModels:
    def test_returns_model_names_for_provider(self) -> None:
        registry = LLMRegistry()
        registry.register(_MODEL_A1, _SETTINGS, _StubProvider())
        registry.register(_MODEL_A2, _SETTINGS, _StubProvider())
        assert set(registry.resolve_models(_PROVIDER_A)) == {"gpt-4o", "gpt-4o-mini"}

    def test_empty_when_provider_unknown(self) -> None:
        registry = LLMRegistry()
        registry.register(_MODEL_A1, _SETTINGS, _StubProvider())
        assert registry.resolve_models("nonexistent") == []

    def test_excludes_other_providers(self) -> None:
        registry = LLMRegistry()
        registry.register(_MODEL_A1, _SETTINGS, _StubProvider())
        registry.register(_MODEL_B1, _SETTINGS, _StubProvider())
        assert registry.resolve_models(_PROVIDER_A) == ["gpt-4o"]
        assert registry.resolve_models(_PROVIDER_B) == ["claude-3-5-sonnet"]


class TestLLMRegistryLLMsProperty:
    def test_empty_when_no_registrations(self) -> None:
        assert LLMRegistry().llms == []

    def test_returns_all_registered_llms(self) -> None:
        registry = LLMRegistry()
        registry.register(_MODEL_A1, _SETTINGS, _StubProvider())
        registry.register(_MODEL_B1, _SETTINGS, _StubProvider())
        assert set(registry.llms) == {_MODEL_A1, _MODEL_B1}

    def test_contains_llm_instances(self) -> None:
        registry = LLMRegistry()
        registry.register(_MODEL_A1, _SETTINGS, _StubProvider())
        assert all(isinstance(llm, LLM) for llm in registry.llms)
