"""Unit tests for ToolService."""

from ai_agent.core.models.tool import Tool, ToolCall, ToolResponse, ToolSchema
from ai_agent.core.registries.tool import ToolRegistry
from ai_agent.core.services.tool import ToolService


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ECHO_SCHEMA = ToolSchema(
    description="Echoes its input.",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
)

_FAIL_SCHEMA = ToolSchema(
    description="Always raises.",
    parameters={"type": "object", "properties": {}},
)


class _EchoTool:
    @property
    def schema(self) -> ToolSchema:
        return _ECHO_SCHEMA

    def execute(self, arguments: dict[str, object]) -> ToolResponse:
        return ToolResponse(content=str(arguments.get("text", "")))


class _FailingTool:
    @property
    def schema(self) -> ToolSchema:
        return _FAIL_SCHEMA

    def execute(self, arguments: dict[str, object]) -> ToolResponse:
        raise RuntimeError("boom")


_ECHO_TOOL = Tool(type="test", name="echo")
_FAIL_TOOL = Tool(type="fail", name="failing")


def _make_registry(*pairs: tuple[Tool, object]) -> ToolRegistry:
    registry = ToolRegistry()
    for tool, impl in pairs:
        registry.register(tool, impl)  # type: ignore[arg-type]
    return registry


def _make_service(*pairs: tuple[Tool, object]) -> ToolService:
    return ToolService(registry=_make_registry(*pairs))


# ---------------------------------------------------------------------------
# ToolService — dispatch
# ---------------------------------------------------------------------------


class TestToolServiceDispatch:
    def test_returns_one_result_per_call(self) -> None:
        svc = _make_service((_ECHO_TOOL, _EchoTool()))
        calls = [
            ToolCall(id="c1", name="echo", arguments={"text": "a"}),
            ToolCall(id="c2", name="echo", arguments={"text": "b"}),
        ]
        results = svc.dispatch(calls)
        assert len(results) == 2

    def test_result_ids_match_call_ids(self) -> None:
        svc = _make_service((_ECHO_TOOL, _EchoTool()))
        calls = [
            ToolCall(id="x1", name="echo", arguments={"text": "hi"}),
            ToolCall(id="x2", name="echo", arguments={"text": "bye"}),
        ]
        results = svc.dispatch(calls)
        assert results[0].id == "x1"
        assert results[1].id == "x2"

    def test_result_content_matches_tool_output(self) -> None:
        svc = _make_service((_ECHO_TOOL, _EchoTool()))
        calls = [ToolCall(id="c1", name="echo", arguments={"text": "hello"})]
        results = svc.dispatch(calls)
        assert results[0].content == "hello"
        assert results[0].is_error is False

    def test_empty_call_list_returns_empty_results(self) -> None:
        svc = _make_service((_ECHO_TOOL, _EchoTool()))
        assert svc.dispatch([]) == []

    def test_order_preserved(self) -> None:
        svc = _make_service((_ECHO_TOOL, _EchoTool()))
        calls = [
            ToolCall(id="c1", name="echo", arguments={"text": "first"}),
            ToolCall(id="c2", name="echo", arguments={"text": "second"}),
            ToolCall(id="c3", name="echo", arguments={"text": "third"}),
        ]
        results = svc.dispatch(calls)
        assert [r.content for r in results] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# ToolService — error handling
# ---------------------------------------------------------------------------


class TestToolServiceErrorHandling:
    def test_unknown_tool_returns_error_result(self) -> None:
        svc = _make_service()
        calls = [ToolCall(id="c1", name="ghost", arguments={})]
        results = svc.dispatch(calls)
        assert results[0].is_error is True
        assert results[0].id == "c1"

    def test_tool_exception_returns_error_result(self) -> None:
        svc = _make_service((_FAIL_TOOL, _FailingTool()))
        calls = [ToolCall(id="c1", name="failing", arguments={})]
        results = svc.dispatch(calls)
        assert results[0].is_error is True
        assert "boom" in results[0].content  # type: ignore[operator]

    def test_error_in_one_call_does_not_affect_others(self) -> None:
        svc = _make_service(
            (_ECHO_TOOL, _EchoTool()),
            (_FAIL_TOOL, _FailingTool()),
        )
        calls = [
            ToolCall(id="c1", name="failing", arguments={}),
            ToolCall(id="c2", name="echo", arguments={"text": "ok"}),
        ]
        results = svc.dispatch(calls)
        assert results[0].is_error is True
        assert results[1].is_error is False
        assert results[1].content == "ok"
