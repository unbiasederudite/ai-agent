"""Base class for all tool implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

from ai_agent.core.models.tool import BaseToolConfig, ToolResponse, ToolSchema


class BaseTool(ABC):
    """Shared construction for all tool implementations."""

    def __init__(self, config: BaseToolConfig) -> None:
        self._config: Final = config

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return the schema this tool exposes to the LLM."""
        ...

    @abstractmethod
    def execute(self, arguments: dict[str, object]) -> ToolResponse:
        """Run the tool with the provided arguments.

        Args:
            arguments: Keyword arguments for the tool.

        Returns:
            ToolResponse with the output or an error flag.
        """
        ...
