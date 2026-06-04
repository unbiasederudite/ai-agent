"""Unit tests for Conversation."""

import pytest
from unittest.mock import patch

from ai_agent.core.services.conversation import Conversation
from ai_agent.core.exceptions import ContextWindowExceededError, UserMessageTooLongError
from ai_agent.core.models.budget import ContextBudget
from ai_agent.core.models.llm import LLM, LLMSettings, LLMUsage
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.run import RunResult, RunSettings
from ai_agent.core.models.tool import Tool, ToolDefinition, ToolResponse, ToolSchema
from ai_agent.core.registries.agent import AgentRegistry
from ai_agent.core.registries.llm import LLMRegistry
from ai_agent.core.registries.tool import ToolRegistry
from ai_agent.core.models.llm import LLMRequest, LLMResponse, FinishReason
from ai_agent.core.models import CompactionResult


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_AGENT = "default"
_LLM = LLM(provider="test", model="test-model")
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=4096)
_RUN_SETTINGS = RunSettings(agent=_AGENT, llm=_LLM, settings=_SETTINGS)


def _run_result(output: str = "reply", inp: int = 10, out: int = 5) -> RunResult:
    usage = LLMUsage(input_tokens=inp, output_tokens=out)
    return RunResult(output=output, turns=1, billed_usage=usage, context_usage=usage)


class _StubAgent:
    """Agent stub; call_count tracks invocations; raises on demand."""

    def __init__(
        self, result: RunResult | None = None, raises: type[Exception] | None = None
    ) -> None:
        self._result = result or _run_result()
        self._raises = raises
        self.calls: list[list[Message]] = []
        self._raise_once = raises is not None

    @property
    def run_settings(self) -> RunSettings:
        return _RUN_SETTINGS

    def run(
        self,
        messages: list[Message],
        provider: object,
        model: str,
        settings: LLMSettings,
        tools: list[ToolDefinition] | None,
    ) -> RunResult:
        self.calls.append(list(messages))
        if self._raise_once and self._raises is not None:
            self._raise_once = False
            raise self._raises("context exceeded")
        return self._result


class _StubProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            message=Message(role=Role.ASSISTANT, content="compacted"),
            finish_reason=FinishReason.STOP,
            usage=LLMUsage(input_tokens=5, output_tokens=3),
        )

    def context_window(self, model: str) -> int:
        return 128_000


class _StubCompaction:
    """Duck-typed CompactionService; records compact() calls."""

    def __init__(self, result: CompactionResult | None = None) -> None:
        self.compact_calls = 0
        self._result = result or CompactionResult(
            messages=[Message(role=Role.SYSTEM, content="summary")]
        )

    def compact(self, messages: list[Message], keep_recent_turns: int = 3) -> CompactionResult:
        self.compact_calls += 1
        return self._result


def _make_registry(agent: _StubAgent) -> AgentRegistry:
    r = AgentRegistry()
    r.register(_AGENT, agent)  # type: ignore[arg-type]
    return r


def _make_llm_registry() -> LLMRegistry:
    r = LLMRegistry()
    r.register(_LLM, _SETTINGS, _StubProvider())  # type: ignore[arg-type]
    return r


def _fresh_budget(should_compact: bool = False) -> ContextBudget:
    if should_compact:
        return ContextBudget(
            context_window=100,
            compaction_threshold=0.5,
            context_usage=LLMUsage(input_tokens=80, output_tokens=0),
        )
    return ContextBudget(context_window=128_000)


_TOOL_A = Tool(type="stub", name="tool_a")
_TOOL_B = Tool(type="stub", name="tool_b")
_SCHEMA = ToolSchema(description="stub", parameters={"type": "object", "properties": {}})


class _StubToolImpl:
    @property
    def schema(self) -> ToolSchema:
        return _SCHEMA

    def execute(self, arguments: dict[str, object]) -> ToolResponse:
        return ToolResponse(content="stub")


def _make_tool_registry(*tools: Tool) -> ToolRegistry:
    r = ToolRegistry()
    for t in tools:
        r.register(t, _StubToolImpl())  # type: ignore[arg-type]
    return r


def _make_conversation(
    agent: _StubAgent | None = None,
    budget: ContextBudget | None = None,
    compaction: _StubCompaction | None = None,
    message_char_limit: int = 1000,
    tool_registry: ToolRegistry | None = None,
) -> Conversation:
    stub_agent = agent or _StubAgent()
    return Conversation(
        agent_registry=_make_registry(stub_agent),
        run_settings=_RUN_SETTINGS,
        llm_registry=_make_llm_registry(),
        tool_registry=tool_registry or ToolRegistry(),
        message_char_limit=message_char_limit,
        context_budget=budget or _fresh_budget(),
        compaction_service=compaction or _StubCompaction(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Message length guard
# ---------------------------------------------------------------------------


class TestMessageCharLimit:
    def test_raises_when_message_exceeds_limit(self) -> None:
        conv = _make_conversation(message_char_limit=5)
        with pytest.raises(UserMessageTooLongError):
            conv.run("123456")

    def test_passes_when_message_equals_limit(self) -> None:
        conv = _make_conversation(message_char_limit=5)
        result = conv.run("12345")
        assert result.output == "reply"

    def test_passes_when_message_is_below_limit(self) -> None:
        conv = _make_conversation(message_char_limit=100)
        result = conv.run("hi")
        assert result.output == "reply"

    def test_error_message_contains_lengths(self) -> None:
        conv = _make_conversation(message_char_limit=3)
        with pytest.raises(UserMessageTooLongError, match="4"):
            conv.run("four")


# ---------------------------------------------------------------------------
# Proactive compaction (should_compact before run)
# ---------------------------------------------------------------------------


class TestProactiveCompaction:
    def test_compact_called_when_budget_should_compact(self) -> None:
        compaction = _StubCompaction()
        conv = _make_conversation(budget=_fresh_budget(should_compact=True), compaction=compaction)
        conv.run("hi")
        assert compaction.compact_calls == 1

    def test_compact_not_called_when_budget_ok(self) -> None:
        compaction = _StubCompaction()
        conv = _make_conversation(budget=_fresh_budget(should_compact=False), compaction=compaction)
        conv.run("hi")
        assert compaction.compact_calls == 0

    def test_budget_reset_after_proactive_compact(self) -> None:
        conv = _make_conversation(budget=_fresh_budget(should_compact=True))
        conv.run("hi")
        assert conv.context_budget.context_usage.input_tokens == 10

    def test_messages_replaced_after_proactive_compact(self) -> None:
        compaction = _StubCompaction(
            CompactionResult(messages=[Message(role=Role.SYSTEM, content="summary")])
        )
        conv = _make_conversation(budget=_fresh_budget(should_compact=True), compaction=compaction)
        conv.run("hi")
        assert conv.messages[0] == Message(role=Role.SYSTEM, content="summary")


# ---------------------------------------------------------------------------
# Reactive compaction (ContextWindowExceededError → compact → retry)
# ---------------------------------------------------------------------------


class TestReactiveCompaction:
    def test_compact_called_on_context_exceeded(self) -> None:
        compaction = _StubCompaction()
        agent = _StubAgent(raises=ContextWindowExceededError)
        conv = _make_conversation(agent=agent, compaction=compaction)
        conv.run("hi")
        assert compaction.compact_calls == 1

    def test_second_agent_call_succeeds_after_compact(self) -> None:
        agent = _StubAgent(raises=ContextWindowExceededError)
        conv = _make_conversation(agent=agent)
        result = conv.run("hi")
        assert result.output == "reply"
        assert len(agent.calls) == 2

    def test_retry_uses_compacted_messages(self) -> None:
        summary = Message(role=Role.SYSTEM, content="compacted history")
        compaction = _StubCompaction(CompactionResult(messages=[summary]))
        agent = _StubAgent(raises=ContextWindowExceededError)
        conv = _make_conversation(agent=agent, compaction=compaction)
        conv.run("hello")
        retry_messages = agent.calls[1]
        assert retry_messages[0] == summary
        assert retry_messages[-1].content == "hello"


# ---------------------------------------------------------------------------
# State update after run
# ---------------------------------------------------------------------------


class TestStateUpdate:
    def test_user_and_assistant_messages_appended(self) -> None:
        conv = _make_conversation()
        conv.run("hello")
        assert conv.messages[-2].role == Role.USER
        assert conv.messages[-1].role == Role.ASSISTANT

    def test_user_message_content_stored(self) -> None:
        conv = _make_conversation()
        conv.run("hello")
        assert conv.messages[-2].content == "hello"

    def test_assistant_output_stored(self) -> None:
        conv = _make_conversation(agent=_StubAgent(_run_result(output="pong")))
        conv.run("ping")
        assert conv.messages[-1].content == "pong"

    def test_billed_usage_accumulates_across_runs(self) -> None:
        conv = _make_conversation(agent=_StubAgent(_run_result(inp=10, out=5)))
        conv.run("first")
        conv.run("second")
        assert conv.billed_usage.input_tokens == 20
        assert conv.billed_usage.output_tokens == 10

    def test_context_budget_updated_after_run(self) -> None:
        conv = _make_conversation(agent=_StubAgent(_run_result(inp=100, out=40)))
        conv.run("hi")
        assert conv.context_budget.context_usage.input_tokens == 100
        assert conv.context_budget.context_usage.output_tokens == 40

    def test_history_grows_with_each_run(self) -> None:
        conv = _make_conversation()
        conv.run("first")
        conv.run("second")
        assert len(conv.messages) == 4


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_messages_cleared(self) -> None:
        conv = _make_conversation()
        conv.run("hi")
        conv.reset()
        assert conv.messages == []

    def test_billed_usage_zeroed(self) -> None:
        conv = _make_conversation()
        conv.run("hi")
        conv.reset()
        assert conv.billed_usage.input_tokens == 0
        assert conv.billed_usage.output_tokens == 0

    def test_budget_usage_zeroed(self) -> None:
        conv = _make_conversation(agent=_StubAgent(_run_result(inp=50, out=20)))
        conv.run("hi")
        conv.reset()
        assert conv.context_budget.context_usage.input_tokens == 0


# ---------------------------------------------------------------------------
# Compaction billing
# ---------------------------------------------------------------------------


class TestCompactionBilling:
    def _compaction_with_usage(self, inp: int, out: int) -> _StubCompaction:
        return _StubCompaction(
            CompactionResult(
                messages=[Message(role=Role.SYSTEM, content="summary")],
                usage=LLMUsage(input_tokens=inp, output_tokens=out),
            )
        )

    def test_proactive_compaction_cost_added_to_billed_usage(self) -> None:
        conv = _make_conversation(
            budget=_fresh_budget(should_compact=True),
            compaction=self._compaction_with_usage(inp=30, out=10),
        )
        conv.run("hi")
        assert conv.billed_usage.input_tokens == 30 + 10  # compaction + run
        assert conv.billed_usage.output_tokens == 10 + 5

    def test_reactive_compaction_cost_added_to_billed_usage(self) -> None:
        conv = _make_conversation(
            agent=_StubAgent(raises=ContextWindowExceededError),
            compaction=self._compaction_with_usage(inp=20, out=8),
        )
        conv.run("hi")
        assert conv.billed_usage.input_tokens == 20 + 10
        assert conv.billed_usage.output_tokens == 8 + 5

    def test_compaction_without_usage_does_not_affect_billing(self) -> None:
        compaction = _StubCompaction(
            CompactionResult(messages=[Message(role=Role.SYSTEM, content="summary")])
        )
        conv = _make_conversation(
            budget=_fresh_budget(should_compact=True),
            compaction=compaction,
        )
        conv.run("hi")
        assert conv.billed_usage.input_tokens == 10
        assert conv.billed_usage.output_tokens == 5


# ---------------------------------------------------------------------------
# sticky — recalibration
# ---------------------------------------------------------------------------


_AGENT_B = "other"
_LLM_B = LLM(provider="test", model="other-model")
_RUN_SETTINGS_B = RunSettings(agent=_AGENT_B, llm=_LLM_B, settings=_SETTINGS)


class _StubAgentB:
    @property
    def run_settings(self) -> RunSettings:
        return _RUN_SETTINGS_B

    def run(
        self,
        messages: list[Message],
        provider: object,
        model: str,
        settings: LLMSettings,
        tools: list[ToolDefinition] | None,
    ) -> RunResult:
        return _run_result()


def _make_conversation_two_agents() -> Conversation:
    agent_a = _StubAgent()
    agent_b = _StubAgentB()
    registry = AgentRegistry()
    registry.register(_AGENT, agent_a)  # type: ignore[arg-type]
    registry.register(_AGENT_B, agent_b)  # type: ignore[arg-type]
    llm_registry = LLMRegistry()
    provider = _StubProvider()
    llm_registry.register(_LLM, _SETTINGS, provider)  # type: ignore[arg-type]
    llm_registry.register(_LLM_B, _SETTINGS, provider)  # type: ignore[arg-type]
    return Conversation(
        agent_registry=registry,
        run_settings=_RUN_SETTINGS,
        llm_registry=llm_registry,
        tool_registry=ToolRegistry(),
        message_char_limit=1000,
        context_budget=ContextBudget(context_window=1000),
        compaction_service=_StubCompaction(),  # type: ignore[arg-type]
    )


def _litellm_patch() -> patch:  # type: ignore[type-arg]
    return patch(
        "ai_agent.core.registries.llm.litellm.get_model_info",
        return_value={"max_input_tokens": 128_000},
    )


class TestStickyRecalibration:
    def test_sticky_agent_recalibrates_budget(self) -> None:
        conv = _make_conversation_two_agents()
        assert conv.context_budget.context_window == 1000
        with _litellm_patch():
            conv.sticky("agent", _AGENT_B)
        assert conv.context_budget.context_window == 128_000

    def test_sticky_agent_updates_run_settings(self) -> None:
        conv = _make_conversation_two_agents()
        with _litellm_patch():
            conv.sticky("agent", _AGENT_B)
        assert conv.active_agent.run_settings.llm.model == "other-model"

    def test_sticky_llm_recalibrates_budget(self) -> None:
        conv = _make_conversation_two_agents()
        assert conv.context_budget.context_window == 1000
        with _litellm_patch():
            conv.sticky("llm", _LLM_B)
        assert conv.context_budget.context_window == 128_000

    def test_sticky_llm_updates_run_settings(self) -> None:
        conv = _make_conversation_two_agents()
        with _litellm_patch():
            conv.sticky("llm", _LLM_B)
        assert conv._run_settings.llm.model == "other-model"  # type: ignore[attr-defined]

    def test_sticky_other_field_does_not_recalibrate(self) -> None:
        conv = _make_conversation_two_agents()
        conv.sticky("settings", LLMSettings(temperature=0.1, max_tokens=512))
        assert conv.context_budget.context_window == 1000


# ---------------------------------------------------------------------------
# _resolve_tools — None vs [] contract
# ---------------------------------------------------------------------------


def _settings_with_tools(tools: list[Tool] | None) -> RunSettings:
    return _RUN_SETTINGS.model_copy(update={"tools": tools})


class TestResolveTools:
    def test_tools_none_returns_all_registered(self) -> None:
        conv = _make_conversation(tool_registry=_make_tool_registry(_TOOL_A, _TOOL_B))
        result = conv._resolve_tools(_settings_with_tools(None))  # noqa: SLF001
        assert result is not None
        assert {d.name for d in result} == {"tool_a", "tool_b"}

    def test_tools_empty_list_returns_none(self) -> None:
        conv = _make_conversation(tool_registry=_make_tool_registry(_TOOL_A))
        result = conv._resolve_tools(_settings_with_tools([]))  # noqa: SLF001
        assert result is None

    def test_tools_subset_returns_only_those(self) -> None:
        conv = _make_conversation(tool_registry=_make_tool_registry(_TOOL_A, _TOOL_B))
        result = conv._resolve_tools(_settings_with_tools([_TOOL_A]))  # noqa: SLF001
        assert result is not None
        assert [d.name for d in result] == ["tool_a"]
