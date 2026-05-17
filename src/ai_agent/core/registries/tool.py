"""Tool registry mapping identifiers to implementations and schemas."""

from __future__ import annotations

from ai_agent.core.exceptions import ToolNotFoundError
from ai_agent.core.models.tool import Tool, ToolDefinition, ToolSchema
from ai_agent.core.protocols.tool import ITool


class ToolRegistry:
    """Maps Tool identifiers to implementations and schemas."""

    def __init__(self) -> None:
        self._callers: dict[str, ITool] = {}
        self._schemas: dict[Tool, ToolSchema] = {}

    def register(self, tool: Tool, schema: ToolSchema, implementation: ITool) -> None:
        """Register an implementation and schema under a Tool key.

        Args:
            tool: Tool identifier (type + name).
            schema: Tool schema exposed to the LLM.
            implementation: ITool adapter.
        """
        self._callers.setdefault(tool.type, implementation)
        self._schemas.setdefault(tool, schema)

    def resolve_implementation(self, type: str) -> ITool:
        """Return the ITool adapter registered under the given tool type.

        Args:
            type: Tool type string.

        Raises:
            ToolNotFoundError: If no implementation is registered for that type.
        """
        try:
            return self._callers[type]
        except KeyError:
            raise ToolNotFoundError(
                f"No implementation for tool type {type!r}. Registered: {sorted(self._callers)}"
            ) from None

    def resolve_definition(self, tool: Tool) -> ToolDefinition:
        """Return a ToolDefinition for the given Tool identity.

        Args:
            tool: Tool identifier.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        try:
            schema = self._schemas[tool]
        except KeyError:
            raise ToolNotFoundError(
                f"No schema for tool type={tool.type!r} name={tool.name!r}."
            ) from None
        return ToolDefinition(name=tool.name, tool_schema=schema)

    def resolve_tools(self, type: str) -> list[str]:
        """Return all tool names registered under the given tool type.

        Args:
            type: Tool type string.
        """
        return [t.name for t in self._schemas if t.type == type]

    @property
    def tools(self) -> list[Tool]:
        """All registered Tool identities."""
        return list(self._schemas)
