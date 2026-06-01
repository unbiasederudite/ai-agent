"""StrategyFactory: resolves a StrategyConfig to a concrete IReasoningStrategy."""

from typing import Final

from ai_agent.core.exceptions import ConfigError
from ai_agent.core.models.strategy import StrategyConfig
from ai_agent.core.protocols.strategy import IReasoningStrategy
from ai_agent.core.services.tool import ToolService
from ai_agent.core.strategies.base import BaseStrategy


class StrategyFactory:
    """Builds an IReasoningStrategy instance from its configuration."""

    def __init__(
        self,
        implementations: dict[str, type[BaseStrategy]],
        tool_service: ToolService,
    ) -> None:
        self._implementations: Final = implementations
        self._tool_service: Final = tool_service

    def build(self, config: StrategyConfig) -> IReasoningStrategy:
        """Instantiate a strategy from its config.

        Args:
            config: Strategy configuration with type discriminator.

        Returns:
            A concrete IReasoningStrategy instance.

        Raises:
            ConfigError: If the strategy type is not in the implementations dict.
        """
        cls = self._implementations.get(config.type)
        if cls is None:
            raise ConfigError(
                f"Unknown strategy type {config.type!r}. Available: {sorted(self._implementations)}"
            )
        return cls(config, self._tool_service)
