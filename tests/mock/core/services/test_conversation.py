"""Mock tests for Conversation.sticky() — requires LiteLLM patching."""

from unittest.mock import patch

from ai_agent.core.models.budget import ContextBudget
from ai_agent.core.models.llm import (
    LLM,
    FinishReason,
    LLMRequest,
    LLMResponse,
    LLMSettings,
    LLMUsage,
)
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.run import RunResult, RunSettings
from ai_agent.core.models.tool import ToolDefinition
from ai_agent.core.registries.agent import AgentRegistry
from ai_agent.core.registries.llm import LLMRegistry
from ai_agent.core.registries.tool import ToolRegistry
from ai_agent.core.services.conversation import Conversation
from ai_agent.core.models import CompactionResult


_AGENT = "default"
_AGENT_B = "other"
_LLM = LLM(provider="test", model="test-model")
_LLM_B = LLM(provider="test", model="other-model")
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=4096)
_RUN_SETTINGS = RunSettings(agent=_AGENT, llm=_LLM, settings=_SETTINGS)
_RUN_SETTINGS_B = RunSettings(agent=_AGENT_B, llm=_LLM_B, settings=_SETTINGS)
_CONTEXT_WINDOW = 128_000


def _run_result() -> RunResult:
    usage = LLMUsage(input_tokens=10, output_tokens=5)
    return RunResult(output="reply", turns=1, billed_usage=usage, context_usage=usage)


class _StubProvider:
    """Minimal stub satisfying ILLMProvider."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            message=Message(role=Role.ASSISTANT, content="ok"),
            finish_reason=FinishReason.STOP,
            usage=LLMUsage(input_tokens=1, output_tokens=1),
        )


class _StubAgent:
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
        return _run_result()


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


class _StubCompaction:
    def compact(self, messages: list[Message], keep_recent_turns: int = 3) -> CompactionResult:
        return CompactionResult(messages=[Message(role=Role.SYSTEM, content="summary")])


def _make_conversation_two_agents() -> Conversation:
    agent_a = _StubAgent()
    agent_b = _StubAgentB()
    registry = AgentRegistry()
    registry.register(_AGENT, agent_a)  # type: ignore[arg-type]
    registry.register(_AGENT_B, agent_b)  # type: ignore[arg-type]
    llm_registry = LLMRegistry()
    provider = _StubProvider()
    llm_registry.register(_LLM, _SETTINGS, provider)
    llm_registry.register(_LLM_B, _SETTINGS, provider)
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
        return_value={"max_input_tokens": _CONTEXT_WINDOW},
    )


class TestStickyRecalibration:
    def test_sticky_agent_recalibrates_budget(self) -> None:
        conv = _make_conversation_two_agents()
        assert conv.context_budget.context_window == 1000
        with _litellm_patch():
            conv.sticky("agent", _AGENT_B)
        assert conv.context_budget.context_window == _CONTEXT_WINDOW

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
        assert conv.context_budget.context_window == _CONTEXT_WINDOW

    def test_sticky_llm_updates_run_settings(self) -> None:
        conv = _make_conversation_two_agents()
        with _litellm_patch():
            conv.sticky("llm", _LLM_B)
        assert conv._run_settings.llm.model == "other-model"  # type: ignore[attr-defined]
