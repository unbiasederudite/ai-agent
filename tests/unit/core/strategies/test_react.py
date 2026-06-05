"""Unit tests for ReActStrategy.step()."""

from __future__ import annotations

import json

from ai_agent.core.models.agent import AgentState, AgentStatus
from ai_agent.core.models.llm import FinishReason, LLMRequest, LLMResponse, LLMSettings, LLMUsage
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.tool import ToolCall, ToolResult
from ai_agent.core.models.strategy import ReActStrategyConfig
from ai_agent.core.strategies.react import ReActStrategy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CONFIG = ReActStrategyConfig()
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=512)
_MODEL = "test-model"


def _usage() -> LLMUsage:
    return LLMUsage(input_tokens=10, output_tokens=5)


def _text_response(content: str = "done") -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=content),
        finish_reason=FinishReason.STOP,
        usage=_usage(),
    )


def _tool_response(tool_calls: list[ToolCall]) -> LLMResponse:
    msg = Message(role=Role.ASSISTANT, content=None, tool_calls=tool_calls)
    return LLMResponse(message=msg, finish_reason=FinishReason.TOOL_CALLS, usage=_usage())


def _make_call(call_id: str = "call_1", name: str = "search") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={"q": "test"})


def _make_result(
    call_id: str = "call_1",
    name: str = "search",
    content: str | dict[str, object] = "result",
    is_error: bool = False,
) -> ToolResult:
    return ToolResult(id=call_id, name=name, content=content, is_error=is_error)


class _StubProvider:
    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.received: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.received.append(request)
        return self._response


class _StubToolService:
    def __init__(self, results: list[ToolResult]) -> None:
        self._results = results
        self.calls: list[list[ToolCall]] = []

    def dispatch(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        self.calls.append(list(tool_calls))
        return self._results


def _make_strategy(
    tool_results: list[ToolResult] | None = None,
    strategy_prompt: str = "default strategy prompt",
) -> tuple[ReActStrategy, _StubToolService]:
    svc = _StubToolService(tool_results or [])
    return ReActStrategy(_CONFIG, svc, strategy_prompt=strategy_prompt), svc  # type: ignore[arg-type]


def _initial_state() -> AgentState:
    return AgentState(messages=[Message(role=Role.USER, content="hello")])


def _make_request() -> LLMRequest:
    return LLMRequest(model=_MODEL, settings=_SETTINGS, messages=_initial_state().messages)


# ---------------------------------------------------------------------------
# No tool calls → COMPLETE
# ---------------------------------------------------------------------------


class TestReActNoToolCalls:
    def test_status_is_complete(self) -> None:
        strategy, _ = _make_strategy()
        result = strategy.step(_initial_state(), _make_request(), _StubProvider(_text_response()))
        assert result.state.status == AgentStatus.COMPLETE

    def test_turn_incremented(self) -> None:
        strategy, _ = _make_strategy()
        result = strategy.step(_initial_state(), _make_request(), _StubProvider(_text_response()))
        assert result.state.turn == 1

    def test_assistant_message_appended(self) -> None:
        strategy, _ = _make_strategy()
        result = strategy.step(
            _initial_state(), _make_request(), _StubProvider(_text_response("done"))
        )
        assert result.state.messages[-1].role == Role.ASSISTANT
        assert result.state.messages[-1].content == "done"

    def test_prior_messages_preserved(self) -> None:
        strategy, _ = _make_strategy()
        result = strategy.step(_initial_state(), _make_request(), _StubProvider(_text_response()))
        assert result.state.messages[0].role == Role.USER

    def test_total_message_count(self) -> None:
        strategy, _ = _make_strategy()
        result = strategy.step(_initial_state(), _make_request(), _StubProvider(_text_response()))
        assert len(result.state.messages) == 2

    def test_response_stored_in_step_result(self) -> None:
        strategy, _ = _make_strategy()
        response = _text_response()
        result = strategy.step(_initial_state(), _make_request(), _StubProvider(response))
        assert result.response is response


# ---------------------------------------------------------------------------
# Tool calls present → RUNNING
# ---------------------------------------------------------------------------


class TestReActWithToolCalls:
    def _run(self, tool_calls: list[ToolCall], tool_results: list[ToolResult]):  # type: ignore[return]
        strategy, svc = _make_strategy(tool_results)
        response = _tool_response(tool_calls)
        result = strategy.step(_initial_state(), _make_request(), _StubProvider(response))
        return result, svc

    def test_status_is_running(self) -> None:
        result, _ = self._run([_make_call()], [_make_result()])
        assert result.state.status == AgentStatus.RUNNING

    def test_turn_incremented(self) -> None:
        result, _ = self._run([_make_call()], [_make_result()])
        assert result.state.turn == 1

    def test_tool_service_receives_all_calls(self) -> None:
        calls = [_make_call("c1", "search"), _make_call("c2", "calc")]
        results = [_make_result("c1", "search"), _make_result("c2", "calc")]
        _, svc = self._run(calls, results)
        assert len(svc.calls[0]) == 2

    def test_assistant_message_appended_before_tool_messages(self) -> None:
        result, _ = self._run([_make_call()], [_make_result()])
        assert result.state.messages[1].role == Role.ASSISTANT

    def test_tool_message_appended_after_assistant(self) -> None:
        result, _ = self._run([_make_call()], [_make_result(content="42")])
        assert result.state.messages[2].role == Role.TOOL
        assert result.state.messages[2].content == "42"

    def test_tool_message_has_correct_call_id(self) -> None:
        result, _ = self._run([_make_call("call_xyz")], [_make_result("call_xyz")])
        assert result.state.messages[2].tool_call_id == "call_xyz"

    def test_multiple_tool_results_all_appended(self) -> None:
        calls = [_make_call("c1", "a"), _make_call("c2", "b")]
        results = [_make_result("c1", "a", "res_a"), _make_result("c2", "b", "res_b")]
        result, _ = self._run(calls, results)
        assert len(result.state.messages) == 4  # user + asst + tool_a + tool_b
        assert result.state.messages[2].content == "res_a"
        assert result.state.messages[3].content == "res_b"

    def test_dict_tool_result_content_serialized_to_json(self) -> None:
        payload: dict[str, object] = {"answer": 42, "unit": "km"}
        result, _ = self._run([_make_call()], [_make_result(content=payload)])
        assert result.state.messages[2].content == json.dumps(payload)

    def test_error_tool_result_still_appended(self) -> None:
        result, _ = self._run([_make_call()], [_make_result(content="boom", is_error=True)])
        assert result.state.messages[2].content == "boom"

    def test_response_stored_in_step_result(self) -> None:
        strategy, _ = _make_strategy([_make_result()])
        response = _tool_response([_make_call()])
        result = strategy.step(_initial_state(), _make_request(), _StubProvider(response))
        assert result.response is response


# ---------------------------------------------------------------------------
# Strategy prompt injection
# ---------------------------------------------------------------------------


class TestReActStrategyPrompt:
    def test_strategy_prompt_appended_to_existing_system(self) -> None:
        provider = _StubProvider(_text_response())
        strategy, _ = _make_strategy(strategy_prompt="use tools wisely")
        state = AgentState(
            messages=[
                Message(role=Role.SYSTEM, content="agent identity"),
                Message(role=Role.USER, content="hello"),
            ]
        )
        request = LLMRequest(model=_MODEL, settings=_SETTINGS, messages=state.messages)
        strategy.step(state, request, provider)
        sent = provider.received[0].messages[0]
        assert sent.role == Role.SYSTEM
        assert sent.content == "agent identity\n\nuse tools wisely"

    def test_strategy_prompt_creates_system_when_absent(self) -> None:
        provider = _StubProvider(_text_response())
        strategy, _ = _make_strategy(strategy_prompt="use tools wisely")
        strategy.step(_initial_state(), _make_request(), provider)
        assert provider.received[0].messages[0].role == Role.SYSTEM
        assert provider.received[0].messages[0].content == "use tools wisely"

    def test_original_user_message_follows_system(self) -> None:
        provider = _StubProvider(_text_response())
        strategy, _ = _make_strategy(strategy_prompt="sp")
        strategy.step(_initial_state(), _make_request(), provider)
        assert provider.received[0].messages[1].role == Role.USER

    def test_custom_strategy_prompt_stored(self) -> None:
        svc = _StubToolService([])
        custom = "my custom instructions"
        s = ReActStrategy(_CONFIG, svc, strategy_prompt=custom)  # type: ignore[arg-type]
        assert s._strategy_prompt == custom  # noqa: SLF001

    def test_default_strategy_prompt_is_non_empty(self) -> None:
        svc = _StubToolService([])
        s = ReActStrategy(_CONFIG, svc)  # type: ignore[arg-type]
        assert s._strategy_prompt  # noqa: SLF001
