"""ITool Protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_agent.core.models.tool import ToolResponse, ToolSchema


@runtime_checkable
class ITool(Protocol):
    """Interface that every tool must satisfy."""

    @property
    def schema(self) -> ToolSchema:
        """Return the schema this tool exposes to the LLM."""
        ...

    def execute(self, arguments: dict[str, object]) -> ToolResponse:
        """Run the tool with the provided arguments.

        Args:
            arguments: Keyword arguments for the tool.

        Returns:
            ToolResponse with the output or an error flag.
        """
        ...
