"""Unit tests for ITool Protocol."""

from ai_agent.core.models.tool import ToolConfig, ToolResponse
from ai_agent.core.protocols import ITool


class _StubTool:
    """Minimal concrete class satisfying ITool."""

    config: ToolConfig = ToolConfig(type="stub", name="stub-tool")

    def execute(self, name: str, arguments: dict[str, object]) -> ToolResponse:
        return ToolResponse(content=f"stub:{name}")


def _accepts_tool(t: ITool) -> str:
    return "ok"


class TestITool:
    """Tests for ITool Protocol structural subtyping."""

    def test_stub_satisfies_protocol(self) -> None:
        assert _accepts_tool(_StubTool()) == "ok"

    def test_execute_returns_tool_response(self) -> None:
        tool = _StubTool()
        result = tool.execute("stub-tool", {"x": 1})
        assert isinstance(result, ToolResponse)

    def test_execute_response_has_content(self) -> None:
        tool = _StubTool()
        result = tool.execute("stub-tool", {})
        assert isinstance(result.content, str)

    def test_config_is_tool_config(self) -> None:
        assert isinstance(_StubTool().config, ToolConfig)

    def test_class_missing_execute_fails_isinstance(self) -> None:
        class _NoExecute:
            config: ToolConfig = ToolConfig(type="stub", name="stub-tool")

        assert not isinstance(_NoExecute(), ITool)

    def test_class_missing_config_fails_isinstance(self) -> None:
        class _NoConfig:
            def execute(self, name: str, arguments: dict[str, object]) -> ToolResponse: ...  # type: ignore[return-value]

        assert not isinstance(_NoConfig(), ITool)
