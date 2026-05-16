"""Unit tests for LLMUsage, LLMSettings, LLMRequest, LLMResponse, FinishReason."""

import pytest

from ai_agent.core.models.llm import (
    FinishReason,
    LLMRequest,
    LLMSettings,
    LLMResponse,
    LLMUsage,
)
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.tool import ToolCall, ToolDefinition, ToolSchema


def _make_llm_settings(**overrides: object) -> LLMSettings:
    defaults: dict[str, object] = {
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    return LLMSettings(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestFinishReason:
    def test_finish_reason_stop(self) -> None:
        assert FinishReason.STOP == "stop"

    def test_finish_reason_tool_calls(self) -> None:
        assert FinishReason.TOOL_CALLS == "tool_calls"

    def test_finish_reason_length(self) -> None:
        assert FinishReason.LENGTH == "length"


class TestLLMUsage:
    def test_constructs_with_input_and_output(self) -> None:
        usage = LLMUsage(input_tokens=10, output_tokens=20)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20

    def test_is_frozen(self) -> None:
        usage = LLMUsage(input_tokens=10, output_tokens=20)
        with pytest.raises(Exception):
            usage.output_tokens = 99  # type: ignore[misc]

    def test_rejects_negative_tokens(self) -> None:
        with pytest.raises(Exception):
            LLMUsage(input_tokens=-1, output_tokens=20)

    def test_zero_tokens_accepted(self) -> None:
        usage = LLMUsage(input_tokens=0, output_tokens=0)
        assert usage.input_tokens == 0


class TestLLMSettings:
    def test_constructs_with_all_fields(self) -> None:
        cfg = _make_llm_settings()
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096

    def test_is_frozen(self) -> None:
        cfg = _make_llm_settings()
        with pytest.raises(Exception):
            cfg.temperature = 0.1  # type: ignore[misc]

    def test_rejects_temperature_above_two(self) -> None:
        with pytest.raises(Exception):
            _make_llm_settings(temperature=2.1)

    def test_has_no_tools_field(self) -> None:
        assert "tools" not in LLMSettings.model_fields

    def test_has_no_model_field(self) -> None:
        assert "model" not in LLMSettings.model_fields

    def test_rejects_zero_max_tokens(self) -> None:
        with pytest.raises(Exception):
            _make_llm_settings(max_tokens=0)


class TestLLMRequest:
    def _make_request(self, **overrides: object) -> LLMRequest:
        msg = Message(role=Role.USER, content="hello")
        defaults: dict[str, object] = {
            "model": "gpt-4o",
            "settings": _make_llm_settings(),
            "messages": [msg],
        }
        return LLMRequest(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_constructs(self) -> None:
        req = self._make_request()
        assert req.model == "gpt-4o"
        assert len(req.messages) == 1

    def test_is_frozen(self) -> None:
        req = self._make_request()
        with pytest.raises(Exception):
            req.messages = []  # type: ignore[misc]

    def test_rejects_empty_messages(self) -> None:
        with pytest.raises(Exception):
            self._make_request(messages=[])

    def test_tools_defaults_to_none(self) -> None:
        req = self._make_request()
        assert req.tools is None

    def test_tools_can_be_set(self) -> None:
        defn = ToolDefinition(
            name="calculator",
            tool_schema=ToolSchema(
                description="Arithmetic.", parameters={"type": "object", "properties": {}}
            ),
        )
        req = self._make_request(tools=[defn])
        assert req.tools is not None
        assert req.tools[0].name == "calculator"

    def test_max_tokens_stored_in_settings(self) -> None:
        req = self._make_request(settings=_make_llm_settings(max_tokens=2048))
        assert req.settings.max_tokens == 2048

    def test_has_no_top_level_max_tokens_field(self) -> None:
        assert "max_tokens" not in LLMRequest.model_fields

    def test_model_is_string(self) -> None:
        req = self._make_request()
        assert isinstance(req.model, str)
        assert req.model == "gpt-4o"


class TestLLMResponse:
    def _usage(self) -> LLMUsage:
        return LLMUsage(input_tokens=10, output_tokens=8)

    def test_constructs_text(self) -> None:
        msg = Message(role=Role.ASSISTANT, content="The answer is 42.")
        resp = LLMResponse(
            message=msg,
            finish_reason=FinishReason.STOP,
            usage=self._usage(),
        )
        assert resp.message.content == "The answer is 42."
        assert resp.finish_reason == FinishReason.STOP

    def test_is_frozen(self) -> None:
        msg = Message(role=Role.ASSISTANT, content="hi")
        resp = LLMResponse(message=msg, finish_reason=FinishReason.STOP, usage=self._usage())
        with pytest.raises(Exception):
            resp.finish_reason = FinishReason.LENGTH  # type: ignore[misc]

    def test_constructs_with_tool_calls(self) -> None:
        tc = ToolCall(id="call_1", name="calculator", arguments={"x": 1})
        msg = Message(role=Role.ASSISTANT, tool_calls=[tc])
        resp = LLMResponse(
            message=msg,
            finish_reason=FinishReason.TOOL_CALLS,
            usage=self._usage(),
        )
        assert resp.finish_reason == FinishReason.TOOL_CALLS
        assert resp.message.tool_calls is not None
        assert resp.message.tool_calls[0].id == "call_1"

    def test_rejects_non_assistant_message(self) -> None:
        with pytest.raises(Exception):
            LLMResponse(
                message=Message(role=Role.USER, content="hi"),
                finish_reason=FinishReason.STOP,
                usage=self._usage(),
            )
