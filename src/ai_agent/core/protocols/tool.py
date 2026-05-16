"""ITool Protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_agent.core.models.tool import ToolConfig, ToolResponse


@runtime_checkable
class ITool(Protocol):
    """Interface that every tool must satisfy."""

    config: ToolConfig

    def execute(self, name: str, arguments: dict[str, object]) -> ToolResponse:
        """Run the tool with the provided arguments.

        Args:
            name: Name of the specific tool to invoke.
            arguments: Keyword arguments for the tool.

        Returns:
            ToolResponse with the output or an error flag.
        """
        ...
