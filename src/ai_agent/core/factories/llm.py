"""LLMFactory: resolves provider names to ILLMProvider instances."""

from __future__ import annotations

from typing import Final

from ai_agent.core.exceptions import ConfigError
from ai_agent.core.protocols.llm import ILLMProvider


class LLMFactory:
    """Resolves provider names to pre-built ILLMProvider instances."""

    def __init__(self, implementations: dict[str, ILLMProvider]) -> None:
        self._implementations: Final = implementations

    def build(self, provider: str) -> ILLMProvider:
        """Resolve a provider by name.

        Args:
            provider: Provider name string.

        Returns:
            ILLMProvider instance.

        Raises:
            ConfigError: If the provider is not registered.
        """
        impl = self._implementations.get(provider)
        if impl is None:
            raise ConfigError(
                f"Unknown provider {provider!r}. Available: {sorted(self._implementations)}"
            )
        return impl
