"""Tool execution service."""

from __future__ import annotations

import logging

from ai_agent.core.models.tool import ToolCall, ToolResult
from ai_agent.core.registries.tool import ToolRegistry

_log = logging.getLogger(__name__)


class ToolService:
    """Executes tool calls against the tool registry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def dispatch(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls and return one ToolResult per call.

        Args:
            tool_calls: Tool calls issued by the LLM.

        Returns:
            ToolResults in the same order as tool_calls.
        """
        return [self._execute(tc) for tc in tool_calls]

    def _execute(self, tc: ToolCall) -> ToolResult:
        tool = next((t for t in self._registry.tools if t.name == tc.name), None)
        if tool is None:
            _log.warning("tool_service.not_found", extra={"tool": tc.name})
            return ToolResult(
                id=tc.id, name=tc.name, content=f"No tool named {tc.name!r}", is_error=True
            )

        try:
            resp = self._registry.resolve_implementation(tool.type).execute(tc.name, tc.arguments)
            return ToolResult(id=tc.id, name=tc.name, content=resp.content, is_error=resp.is_error)
        except Exception as exc:  # noqa: BLE001
            _log.warning("tool_service.execute_failed", extra={"tool": tc.name, "error": str(exc)})
            return ToolResult(id=tc.id, name=tc.name, content=str(exc), is_error=True)
