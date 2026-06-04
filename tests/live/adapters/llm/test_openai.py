"""Live tests for OpenAI completion adapter.

Required runtime: OpenAI API reachable.
Required env vars: OPENAI_API_KEY.
"""

from __future__ import annotations

import os

import openai
import pytest

from ai_agent.adapters.llm.openai import OpenAIAdapter
from ai_agent.core.models.llm import FinishReason, LLMRequest, LLMResponse, LLMSettings
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.tool import ToolDefinition, ToolSchema

pytestmark = pytest.mark.skipif(not os.getenv("LIVE_TESTS"), reason="opt-in: set LIVE_TESTS=1")

_MODEL = "gpt-4o-mini"
_SETTINGS = LLMSettings(temperature=0.0, max_tokens=64)


def _make_adapter() -> OpenAIAdapter:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIAdapter(openai.OpenAI(api_key=api_key))


class TestOpenAIAdapterLive:
    def test_complete_returns_assistant_message(self) -> None:
        adapter = _make_adapter()
        request = LLMRequest(
            model=_MODEL,
            settings=_SETTINGS,
            messages=[Message(role=Role.USER, content="Reply with exactly one word: hello")],
        )
        result = adapter.complete(request)
        assert isinstance(result, LLMResponse)
        assert result.message.role == Role.ASSISTANT
        assert result.message.content is not None
        assert result.finish_reason == FinishReason.STOP

    def test_complete_records_token_usage(self) -> None:
        adapter = _make_adapter()
        request = LLMRequest(
            model=_MODEL,
            settings=_SETTINGS,
            messages=[Message(role=Role.USER, content="Say: ok")],
        )
        result = adapter.complete(request)
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0

    def test_complete_with_system_prompt(self) -> None:
        adapter = _make_adapter()
        request = LLMRequest(
            model=_MODEL,
            settings=_SETTINGS,
            messages=[
                Message(role=Role.SYSTEM, content="You are a concise assistant."),
                Message(role=Role.USER, content="Say: ok"),
            ],
        )
        result = adapter.complete(request)
        assert result.message.role == Role.ASSISTANT
        assert result.message.content is not None

    def test_complete_with_tools_returns_tool_call(self) -> None:
        adapter = _make_adapter()
        schema = ToolSchema(
            description="Get the current weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        )
        tool = ToolDefinition(name="get_weather", tool_schema=schema)
        request = LLMRequest(
            model=_MODEL,
            settings=_SETTINGS,
            messages=[Message(role=Role.USER, content="What is the weather in Paris?")],
            tools=[tool],
        )
        result = adapter.complete(request)
        assert result.finish_reason == FinishReason.TOOL_CALLS
        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "get_weather"
        assert "city" in result.tool_calls[0].arguments
