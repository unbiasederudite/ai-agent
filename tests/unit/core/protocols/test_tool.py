"""Unit tests for ITool Protocol."""

from ai_agent.core.models.tool import ToolResponse, ToolSchema
from ai_agent.core.protocols import ITool


_SCHEMA = ToolSchema(description="stub tool", parameters={"type": "object", "properties": {}})


class _StubTool:
    """Minimal concrete class satisfying ITool."""

    @property
    def schema(self) -> ToolSchema:
        return _SCHEMA

    def execute(self, arguments: dict[str, object]) -> ToolResponse:
        return ToolResponse(content="stub")


def _accepts_tool(t: ITool) -> str:
    return "ok"


class TestITool:
    """Tests for ITool Protocol structural subtyping."""

    def test_stub_satisfies_protocol(self) -> None:
        assert _accepts_tool(_StubTool()) == "ok"

    def test_schema_returns_tool_schema(self) -> None:
        assert isinstance(_StubTool().schema, ToolSchema)

    def test_execute_returns_tool_response(self) -> None:
        result = _StubTool().execute({"x": 1})
        assert isinstance(result, ToolResponse)

    def test_execute_response_has_content(self) -> None:
        result = _StubTool().execute({})
        assert isinstance(result.content, str)

    def test_class_missing_execute_fails_isinstance(self) -> None:
        class _NoExecute:
            @property
            def schema(self) -> ToolSchema:
                return _SCHEMA

        assert not isinstance(_NoExecute(), ITool)

    def test_class_missing_schema_fails_isinstance(self) -> None:
        class _NoSchema:
            def execute(self, arguments: dict[str, object]) -> ToolResponse: ...  # type: ignore[return-value]

        assert not isinstance(_NoSchema(), ITool)
