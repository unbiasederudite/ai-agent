"""Unit tests for Tool, ToolSchema, ToolContext, ToolCall, ToolResponse, and ToolResult models."""

import pytest

from ai_agent.core.models.tool import ToolCall, ToolContext, ToolResponse, ToolResult


class TestToolContext:
    """Tests for the ToolContext model."""

    def test_tool_context_constructs(self) -> None:
        ctx = ToolContext(id="c1", name="calculator")
        assert ctx.id == "c1"
        assert ctx.name == "calculator"

    def test_tool_context_is_frozen(self) -> None:
        ctx = ToolContext(id="c1", name="calculator")
        with pytest.raises(Exception):
            ctx.name = "other"  # type: ignore[misc]


class TestToolCall:
    """Tests for the ToolCall model."""

    def test_tool_call_constructs(self) -> None:
        tc = ToolCall(id="call_1", name="calculator", arguments={"x": 1, "y": 2})
        assert tc.id == "call_1"
        assert tc.name == "calculator"
        assert tc.arguments == {"x": 1, "y": 2}

    def test_tool_call_is_tool_context(self) -> None:
        tc = ToolCall(id="call_1", name="calculator", arguments={})
        assert isinstance(tc, ToolContext)

    def test_tool_call_is_frozen(self) -> None:
        tc = ToolCall(id="call_1", name="calculator", arguments={})
        with pytest.raises(Exception):
            tc.name = "other"  # type: ignore[misc]

    def test_tool_call_empty_arguments_allowed(self) -> None:
        tc = ToolCall(id="call_2", name="ping", arguments={})
        assert tc.arguments == {}

    def test_tool_call_nested_arguments_allowed(self) -> None:
        tc = ToolCall(
            id="call_3", name="search", arguments={"query": "AI", "filters": {"lang": "en"}}
        )
        assert tc.arguments["filters"] == {"lang": "en"}


class TestToolResponse:
    """Tests for the ToolResponse model."""

    def test_tool_response_constructs(self) -> None:
        resp = ToolResponse(content="42")
        assert resp.content == "42"

    def test_tool_response_is_frozen(self) -> None:
        resp = ToolResponse(content="42")
        with pytest.raises(Exception):
            resp.content = "modified"  # type: ignore[misc]

    def test_tool_response_error_defaults_to_false(self) -> None:
        resp = ToolResponse(content="42")
        assert resp.is_error is False

    def test_tool_response_error_can_be_set(self) -> None:
        resp = ToolResponse(content="division by zero", is_error=True)
        assert resp.is_error is True

    def test_tool_response_content_accepts_dict(self) -> None:
        resp = ToolResponse(content={"results": ["a", "b"], "count": 2})
        assert isinstance(resp.content, dict)

    def test_tool_response_has_no_id_field(self) -> None:
        assert "id" not in ToolResponse.model_fields

    def test_tool_response_has_no_name_field(self) -> None:
        assert "name" not in ToolResponse.model_fields


class TestToolResult:
    """Tests for the ToolResult model."""

    def _make(self, **overrides: object) -> ToolResult:
        defaults: dict[str, object] = {"id": "c1", "name": "calc", "content": "42"}
        return ToolResult(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_tool_result_constructs(self) -> None:
        tr = self._make()
        assert tr.id == "c1"
        assert tr.name == "calc"
        assert tr.content == "42"

    def test_tool_result_is_tool_context(self) -> None:
        assert isinstance(self._make(), ToolContext)

    def test_tool_result_is_tool_response(self) -> None:
        assert isinstance(self._make(), ToolResponse)

    def test_tool_result_is_frozen(self) -> None:
        tr = self._make()
        with pytest.raises(Exception):
            tr.content = "modified"  # type: ignore[misc]

    def test_tool_result_error_defaults_to_false(self) -> None:
        assert self._make().is_error is False

    def test_tool_result_error_can_be_set(self) -> None:
        tr = self._make(content="division by zero", is_error=True)
        assert tr.is_error is True

    def test_tool_result_content_accepts_dict(self) -> None:
        tr = self._make(content={"results": ["a", "b"], "count": 2})
        assert isinstance(tr.content, dict)
        assert tr.content["count"] == 2  # type: ignore[index]
