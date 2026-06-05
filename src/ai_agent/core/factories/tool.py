"""ToolFactory: resolves a BaseToolConfig to a concrete ITool."""

from typing import Final

from ai_agent.core.exceptions import ConfigError
from ai_agent.core.models.tool import BaseToolConfig
from ai_agent.core.protocols.tool import ITool
from ai_agent.core.tools.base import BaseTool


class ToolFactory:
    """Builds an ITool instance from its configuration."""

    def __init__(self, implementations: dict[str, type[BaseTool]]) -> None:
        self._implementations: Final = implementations

    def build(self, config: BaseToolConfig) -> ITool:
        """Instantiate a tool from its config.

        Args:
            config: Tool configuration with type discriminator.

        Returns:
            A concrete ITool instance.

        Raises:
            ConfigError: If the tool type is not in the implementations dict.
        """
        cls = self._implementations.get(config.type)
        if cls is None:
            raise ConfigError(
                f"Unknown tool type {config.type!r}. Available: {sorted(self._implementations)}"
            )
        return cls(config)
