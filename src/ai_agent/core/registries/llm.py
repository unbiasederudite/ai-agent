"""LLM registry mapping identifiers to providers and settings."""

from __future__ import annotations

import litellm

from ai_agent.core.exceptions import ConfigError, ProviderNotFoundError
from ai_agent.core.models.llm import LLM, LLMSettings
from ai_agent.core.protocols.llm import ILLMProvider


class LLMRegistry:
    """Maps LLM identifiers to providers and sampling defaults."""

    def __init__(self) -> None:
        self._callers: dict[str, ILLMProvider] = {}
        self._settings: dict[LLM, LLMSettings] = {}

    def register(self, llm: LLM, settings: LLMSettings, implementation: ILLMProvider) -> None:
        """Register a provider and sampling defaults under an LLM key.

        Args:
            llm: LLM identifier.
            settings: Default sampling parameters.
            implementation: Provider adapter.
        """
        self._callers.setdefault(llm.provider, implementation)
        self._settings.setdefault(llm, settings)

    def resolve_implementation(self, provider: str) -> ILLMProvider:
        """Return the provider adapter registered under the given provider name.

        Args:
            provider: Provider name string.

        Raises:
            ProviderNotFoundError: If the provider is not registered.
        """
        try:
            return self._callers[provider]
        except KeyError:
            raise ProviderNotFoundError(
                f"Unknown provider {provider!r}. Registered: {sorted(self._callers)}"
            ) from None

    def resolve_settings(self, llm: LLM) -> LLMSettings:
        """Return the sampling defaults registered for the given LLM.

        Args:
            llm: LLM identifier.

        Raises:
            ProviderNotFoundError: If the LLM is not registered.
        """
        try:
            return self._settings[llm]
        except KeyError:
            registered = [(k.provider, k.model) for k in self._settings]
            raise ProviderNotFoundError(
                f"Unknown LLM provider={llm.provider!r} model={llm.model!r}. "
                f"Registered: {registered}"
            ) from None

    def resolve_models(self, provider: str) -> list[str]:
        """Return all model names registered under the given provider.

        Args:
            provider: Provider name string.
        """
        return [llm.model for llm in self._settings if llm.provider == provider]

    def resolve_context_window(self, llm: LLM) -> int:
        """Return the context window size for the given LLM via LiteLLM.

        Args:
            llm: LLM identifier.

        Raises:
            ProviderNotFoundError: If the LLM is not registered.
            ConfigError: If LiteLLM cannot determine the context window.
        """
        if llm not in self._settings:
            registered = [(k.provider, k.model) for k in self._settings]
            raise ProviderNotFoundError(
                f"Unknown LLM provider={llm.provider!r} model={llm.model!r}. "
                f"Registered: {registered}"
            )
        try:
            info = litellm.get_model_info(f"{llm.provider}/{llm.model}")
        except Exception as exc:
            raise ConfigError(
                f"Cannot determine context window for {llm.provider}/{llm.model}: {exc}"
            ) from exc
        max_tokens = info.get("max_input_tokens")
        if not max_tokens:
            raise ConfigError(
                f"LiteLLM returned no max_input_tokens for {llm.provider}/{llm.model}"
            )
        return int(max_tokens)

    @property
    def llms(self) -> list[LLM]:
        """All registered LLMs."""
        return list(self._settings)
