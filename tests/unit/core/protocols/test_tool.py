"""Unit tests for ITool Protocol."""

from ai_agent.core.models.tool import ToolResponse
from ai_agent.core.protocols import ITool


class _StubTool:
    """Minimal concrete class satisfying ITool."""

    def execute(self, name: str, arguments: dict[str, object]) -> ToolResponse:
        return ToolResponse(content=f"stub:{name}")


def _accepts_tool(t: ITool) -> str:
    return "ok"


class TestITool:
    """Tests for ITool Protocol structural subtyping."""

    def test_stub_satisfies_protocol(self) -> None:
        assert _accepts_tool(_StubTool()) == "ok"

    def test_execute_returns_tool_response(self) -> None:
        result = _StubTool().execute("stub-tool", {"x": 1})
        assert isinstance(result, ToolResponse)

    def test_execute_response_has_content(self) -> None:
        result = _StubTool().execute("stub-tool", {})
        assert isinstance(result.content, str)

    def test_class_missing_execute_fails_isinstance(self) -> None:
        class _NoExecute:
            pass

        assert not isinstance(_NoExecute(), ITool)
