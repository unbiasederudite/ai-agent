"""Unit tests for RunSettings and RunResult."""

import pytest

from ai_agent.core.models.llm import LLM, LLMSettings, LLMUsage
from ai_agent.core.models.run import RunResult, RunSettings
from ai_agent.core.models.tool import Tool


def _usage(inp: int = 10, out: int = 5) -> LLMUsage:
    return LLMUsage(input_tokens=inp, output_tokens=out)


def _make_run_settings(**overrides: object) -> RunSettings:
    defaults: dict[str, object] = {
        "llm": LLM(provider="openai", model="gpt-4o"),
        "settings": LLMSettings(temperature=0.7, max_tokens=4096),
    }
    return RunSettings(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestRunResult:
    def test_constructs_with_required_fields(self) -> None:
        result = RunResult(output="hello", turns=2, usage=_usage())
        assert result.output == "hello"
        assert result.turns == 2

    def test_usage_field_is_stored(self) -> None:
        u = _usage(inp=100, out=50)
        result = RunResult(output="hi", turns=1, usage=u)
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50

    def test_turns_zero_is_valid(self) -> None:
        result = RunResult(output="hi", turns=0, usage=_usage())
        assert result.turns == 0

    def test_negative_turns_raises(self) -> None:
        with pytest.raises(Exception):
            RunResult(output="hi", turns=-1, usage=_usage())

    def test_is_frozen(self) -> None:
        result = RunResult(output="hi", turns=1, usage=_usage())
        with pytest.raises(Exception):
            result.output = "changed"  # type: ignore[misc]


class TestRunSettings:
    def test_constructs_with_required_fields(self) -> None:
        rs = _make_run_settings()
        assert rs.llm.provider == "openai"
        assert rs.llm.model == "gpt-4o"
        assert rs.settings.temperature == 0.7
        assert rs.settings.max_tokens == 4096

    def test_tools_defaults_to_none(self) -> None:
        rs = _make_run_settings()
        assert rs.tools is None

    def test_tools_can_be_empty_list(self) -> None:
        rs = _make_run_settings(tools=[])
        assert rs.tools == []

    def test_tools_can_be_tool_list(self) -> None:
        tools = [Tool(type="mcp", name="search"), Tool(type="mcp", name="calc")]
        rs = _make_run_settings(tools=tools)
        assert rs.tools == tools

    def test_is_frozen(self) -> None:
        rs = _make_run_settings()
        with pytest.raises(Exception):
            rs.llm = LLM(provider="x", model="y")  # type: ignore[misc]

    def test_sticky_model_change_via_model_copy(self) -> None:
        rs = _make_run_settings()
        new_llm = rs.llm.model_copy(update={"model": "o3"})
        rs2 = rs.model_copy(update={"llm": new_llm})
        assert rs2.llm.model == "o3"
        assert rs.llm.model == "gpt-4o"  # original unchanged

    def test_sticky_tools_change_via_model_copy(self) -> None:
        rs = _make_run_settings(tools=None)
        rs2 = rs.model_copy(update={"tools": ["search"]})
        assert rs2.tools == ["search"]
        assert rs.tools is None

    def test_requires_llm(self) -> None:
        with pytest.raises(Exception):
            RunSettings(settings=LLMSettings(temperature=0.7, max_tokens=4096))  # type: ignore[call-arg]

    def test_requires_settings(self) -> None:
        with pytest.raises(Exception):
            RunSettings(llm=LLM(provider="openai", model="gpt-4o"))  # type: ignore[call-arg]

    def test_max_tokens_absent_at_top_level(self) -> None:
        assert "max_tokens" not in RunSettings.model_fields

    def test_temperature_absent_at_top_level(self) -> None:
        assert "temperature" not in RunSettings.model_fields
