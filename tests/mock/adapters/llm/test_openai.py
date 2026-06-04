"""Mock tests for OpenAIAdapter.complete()."""

from __future__ import annotations

from unittest.mock import MagicMock

import openai
import pytest
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function as ToolCallFn

from ai_agent.adapters.llm.openai import OpenAIAdapter
from ai_agent.core.exceptions import (
    AuthenticationError,
    CompletionError,
    ContextWindowExceededError,
    RateLimitError,
)
from ai_agent.core.models.llm import FinishReason, LLMRequest, LLMResponse, LLMSettings
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.tool import ToolCall, ToolDefinition, ToolSchema


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SETTINGS = LLMSettings(temperature=0.7, max_tokens=512)
_MODEL = "gpt-4o"


def _make_adapter() -> tuple[OpenAIAdapter, MagicMock]:
    client: MagicMock = MagicMock(spec=openai.OpenAI)
    return OpenAIAdapter(client), client


def _make_request(
    messages: list[Message] | None = None,
    tools: list[ToolDefinition] | None = None,
) -> LLMRequest:
    if messages is None:
        messages = [Message(role=Role.USER, content="hello")]
    return LLMRequest(model=_MODEL, settings=_SETTINGS, messages=messages, tools=tools)


def _make_completion(
    content: str | None = "reply",
    finish_reason: str = "stop",
    tool_calls: list[ChatCompletionMessageToolCall] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a minimal ChatCompletion mock."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    return completion


def _make_tool_call(
    call_id: str = "call_abc",
    name: str = "my_tool",
    arguments: str = '{"key": "val"}',
) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id, type="function", function=ToolCallFn(name=name, arguments=arguments)
    )


def _status_error(
    exc_cls: type,
    message: str = "error",
    code: str | None = None,
) -> Exception:
    """Create an openai APIStatusError subclass with an optional error code."""
    body: dict[str, object] | None = {"code": code} if code is not None else None
    return exc_cls(message, response=MagicMock(), body=body)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestCompleteHappyPath:
    def test_returns_llm_response(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        assert isinstance(adapter.complete(_make_request()), LLMResponse)

    def test_response_role_is_assistant(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        assert adapter.complete(_make_request()).message.role == Role.ASSISTANT

    def test_response_content_mapped(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion(content="hello back")
        assert adapter.complete(_make_request()).message.content == "hello back"

    def test_usage_tokens_mapped(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion(
            prompt_tokens=20, completion_tokens=7
        )
        result = adapter.complete(_make_request())
        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 7

    def test_model_forwarded_to_openai(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        adapter.complete(_make_request())
        assert client.chat.completions.create.call_args.kwargs["model"] == _MODEL

    def test_temperature_forwarded_to_openai(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        adapter.complete(_make_request())
        assert (
            client.chat.completions.create.call_args.kwargs["temperature"] == _SETTINGS.temperature
        )

    def test_max_tokens_forwarded_to_openai(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        adapter.complete(_make_request())
        assert client.chat.completions.create.call_args.kwargs["max_tokens"] == _SETTINGS.max_tokens


# ---------------------------------------------------------------------------
# Finish reason mapping
# ---------------------------------------------------------------------------


class TestFinishReasonMapping:
    @pytest.mark.parametrize(
        ("openai_reason", "expected"),
        [
            pytest.param("stop", FinishReason.STOP, id="stop"),
            pytest.param("tool_calls", FinishReason.TOOL_CALLS, id="tool_calls"),
            pytest.param("function_call", FinishReason.TOOL_CALLS, id="function_call_legacy_alias"),
            pytest.param("length", FinishReason.LENGTH, id="length"),
            pytest.param("max_tokens", FinishReason.LENGTH, id="max_tokens_legacy_alias"),
            pytest.param("content_filter", FinishReason.STOP, id="content_filter_maps_to_stop"),
        ],
    )
    def test_finish_reason_mapped(self, openai_reason: str, expected: FinishReason) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion(finish_reason=openai_reason)
        assert adapter.complete(_make_request()).finish_reason == expected


# ---------------------------------------------------------------------------
# Message mapping
# ---------------------------------------------------------------------------


class TestMessageMapping:
    def test_user_message_mapped(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        adapter.complete(_make_request(messages=[Message(role=Role.USER, content="hi")]))
        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hi"

    def test_system_message_mapped(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        adapter.complete(
            _make_request(
                messages=[
                    Message(role=Role.SYSTEM, content="be helpful"),
                    Message(role=Role.USER, content="hi"),
                ]
            )
        )
        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "be helpful"

    def test_assistant_message_mapped(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        adapter.complete(
            _make_request(
                messages=[
                    Message(role=Role.USER, content="hi"),
                    Message(role=Role.ASSISTANT, content="hello"),
                    Message(role=Role.USER, content="bye"),
                ]
            )
        )
        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "hello"

    def test_tool_message_mapped(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        adapter.complete(
            _make_request(
                messages=[
                    Message(role=Role.USER, content="hi"),
                    Message(role=Role.TOOL, content="result", tool_call_id="call_1"),
                ]
            )
        )
        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        assert msgs[1]["role"] == "tool"
        assert msgs[1]["tool_call_id"] == "call_1"
        assert msgs[1]["content"] == "result"

    def test_assistant_message_with_tool_calls_mapped(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        tc = ToolCall(id="call_1", name="fn", arguments={"x": 1})
        adapter.complete(
            _make_request(
                messages=[
                    Message(role=Role.USER, content="run fn"),
                    Message(role=Role.ASSISTANT, content=None, tool_calls=[tc]),
                ]
            )
        )
        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        assert msgs[1]["role"] == "assistant"
        assert "tool_calls" in msgs[1]
        assert msgs[1]["tool_calls"][0]["id"] == "call_1"
        assert msgs[1]["tool_calls"][0]["function"]["name"] == "fn"


# ---------------------------------------------------------------------------
# Tool request handling
# ---------------------------------------------------------------------------


class TestToolHandling:
    def test_tools_key_absent_when_request_tools_is_none(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        adapter.complete(_make_request(tools=None))
        assert "tools" not in client.chat.completions.create.call_args.kwargs

    def test_tools_forwarded_when_provided(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion()
        schema = ToolSchema(description="a tool", parameters={"type": "object", "properties": {}})
        tool = ToolDefinition(name="my_tool", tool_schema=schema)
        adapter.complete(_make_request(tools=[tool]))
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "tools" in kwargs
        assert kwargs["tools"][0]["function"]["name"] == "my_tool"
        assert kwargs["tools"][0]["function"]["description"] == "a tool"


# ---------------------------------------------------------------------------
# Tool call response mapping
# ---------------------------------------------------------------------------


class TestToolCallResponseMapping:
    def test_tool_calls_in_response_mapped(self) -> None:
        adapter, client = _make_adapter()
        tc = _make_tool_call(call_id="call_xyz", name="search", arguments='{"q": "cats"}')
        client.chat.completions.create.return_value = _make_completion(tool_calls=[tc])
        result = adapter.complete(_make_request())
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_xyz"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "cats"}

    def test_no_tool_calls_when_response_has_none(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_completion(tool_calls=None)
        assert adapter.complete(_make_request()).tool_calls is None

    def test_malformed_tool_call_json_raises_completion_error(self) -> None:
        adapter, client = _make_adapter()
        tc = _make_tool_call(arguments="not { valid json")
        client.chat.completions.create.return_value = _make_completion(tool_calls=[tc])
        with pytest.raises(CompletionError):
            adapter.complete(_make_request())


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    def test_authentication_error_raises_auth_error(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.side_effect = _status_error(openai.AuthenticationError)
        with pytest.raises(AuthenticationError):
            adapter.complete(_make_request())

    def test_rate_limit_error_raises_rate_limit_error(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.side_effect = _status_error(openai.RateLimitError)
        with pytest.raises(RateLimitError):
            adapter.complete(_make_request())

    def test_connection_error_raises_completion_error(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.side_effect = openai.APIConnectionError(request=MagicMock())
        with pytest.raises(CompletionError):
            adapter.complete(_make_request())

    def test_timeout_error_raises_completion_error(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.side_effect = openai.APITimeoutError(request=MagicMock())
        with pytest.raises(CompletionError):
            adapter.complete(_make_request())

    def test_context_length_exceeded_raises_context_window_error(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.side_effect = _status_error(
            openai.BadRequestError, code="context_length_exceeded"
        )
        with pytest.raises(ContextWindowExceededError):
            adapter.complete(_make_request())

    def test_other_bad_request_raises_completion_error(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.side_effect = _status_error(
            openai.BadRequestError, code="invalid_request_error"
        )
        with pytest.raises(CompletionError):
            adapter.complete(_make_request())

    def test_generic_api_error_raises_completion_error(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.side_effect = openai.InternalServerError(
            "server error", response=MagicMock(), body=None
        )
        with pytest.raises(CompletionError):
            adapter.complete(_make_request())
