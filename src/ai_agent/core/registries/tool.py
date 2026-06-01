"""Tool registry mapping Tool identities to implementations."""

from __future__ import annotations

from ai_agent.core.exceptions import ToolNotFoundError
from ai_agent.core.models.tool import Tool, ToolDefinition
from ai_agent.core.protocols.tool import ITool


class ToolRegistry:
    """Maps Tool identifiers to ITool implementations."""

    def __init__(self) -> None:
        self._callers: dict[Tool, ITool] = {}

    def register(self, tool: Tool, implementation: ITool) -> None:
        """Register an implementation under a Tool key. First registration wins.

        Args:
            tool: Tool identifier (type + name).
            implementation: ITool instance.
        """
        self._callers.setdefault(tool, implementation)

    def resolve_implementation(self, tool: Tool) -> ITool:
        """Return the ITool implementation registered under the given Tool.

        Args:
            tool: Tool identifier.

        Raises:
            ToolNotFoundError: If no implementation is registered for that tool.
        """
        impl = self._callers.get(tool)
        if impl is None:
            raise ToolNotFoundError(
                f"No implementation for tool type={tool.type!r} name={tool.name!r}."
            )
        return impl

    def resolve_definition(self, tool: Tool) -> ToolDefinition:
        """Return a ToolDefinition for the given Tool identity.

        Args:
            tool: Tool identifier.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        impl = self.resolve_implementation(tool)
        return ToolDefinition(name=tool.name, tool_schema=impl.schema)

    def resolve_tools(self, type: str) -> list[str]:
        """Return all tool names registered under the given tool type.

        Args:
            type: Tool type string.
        """
        return [k.name for k in self._callers if k.type == type]

    @property
    def tools(self) -> list[Tool]:
        """All registered Tool identities."""
        return list(self._callers)
