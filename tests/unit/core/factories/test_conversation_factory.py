"""Unit tests for ConversationFactory."""

import pytest
from unittest.mock import patch

from ai_agent.core.services.conversation import Conversation
from ai_agent.core.exceptions import ConfigError
from ai_agent.core.factories.conversation import ConversationFactory
from ai_agent.core.models.agent import AgentState, StepResult
from ai_agent.core.models.config import (
    CompactionConfig,
    ConversationConfig,
    LLMConfig,
    LLMProviderConfig,
)
from ai_agent.core.models.llm import (
    LLM,
    FinishReason,
    LLMRequest,
    LLMResponse,
    LLMSettings,
    LLMUsage,
)
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.strategy import ReActStrategyConfig
from ai_agent.core.models.tool import Tool, BaseToolConfig, ToolResponse, ToolSchema
from ai_agent.core.protocols.llm import ILLMProvider
from ai_agent.core.strategies.base import BaseStrategy
from ai_agent.core.tools.base import BaseTool


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_PROVIDER = "test"
_MODEL = "test-model"
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=4096)
_LLM = LLM(provider=_PROVIDER, model=_MODEL)


class _StubProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            message=Message(role=Role.ASSISTANT, content="ok"),
            finish_reason=FinishReason.STOP,
            usage=LLMUsage(input_tokens=1, output_tokens=1),
        )


class _StubStrategy(BaseStrategy):
    def step(self, state: AgentState, request: LLMRequest, provider: ILLMProvider) -> StepResult:
        raise NotImplementedError


_TOOL_SCHEMA = ToolSchema(description="stub", parameters={"type": "object", "properties": {}})


class _StubTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return _TOOL_SCHEMA

    def execute(self, arguments: dict[str, object]) -> ToolResponse:
        return ToolResponse(content="stub")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: object) -> ConversationConfig:
    defaults: dict[str, object] = {
        "llm_registry": [
            LLMProviderConfig(
                provider=_PROVIDER,
                models=[LLMConfig(model=_MODEL, settings=_SETTINGS)],
            )
        ],
        "agent_registry": [_make_agent_config()],
        "default_agent": "default",
        "compaction": CompactionConfig(
            llm=_LLM,
            settings=LLMSettings(temperature=0.3, max_tokens=2048),
            threshold=0.8,
        ),
    }
    return ConversationConfig(**{**defaults, **overrides})  # type: ignore[arg-type]


def _make_agent_config(name: str = "default") -> object:
    from ai_agent.core.models.agent import AgentConfig

    return AgentConfig(
        name=name,
        description="Test agent.",
        llm=_LLM,
        settings=_SETTINGS,
        strategy=ReActStrategyConfig(),
        tools=[],
    )


def _factory(
    providers: dict[str, object] | None = None,
    tools: dict[str, type[BaseTool]] | None = None,
    strategies: dict[str, type[BaseStrategy]] | None = None,
) -> ConversationFactory:
    return ConversationFactory(
        llm_implementations=providers or {_PROVIDER: _StubProvider()},  # type: ignore[arg-type]
        tool_implementations=tools or {},
        strategy_implementations=strategies or {"react": _StubStrategy},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _litellm_patch() -> patch:  # type: ignore[type-arg]
    return patch(
        "ai_agent.core.registries.llm.litellm.get_model_info",
        return_value={"max_input_tokens": 128_000},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConversationFactoryBuild:
    def test_returns_conversation_instance(self) -> None:
        with _litellm_patch():
            assert isinstance(_factory().build(_make_config()), Conversation)

    def test_unknown_provider_raises_config_error(self) -> None:
        with pytest.raises(ConfigError, match=_PROVIDER):
            _factory(providers={"other": _StubProvider()}).build(_make_config())

    def test_unknown_tool_type_raises_config_error(self) -> None:
        config = _make_config(
            tool_registry=[BaseToolConfig(type="unknown", name="x")],
        )
        with pytest.raises(ConfigError, match="unknown"):
            _factory().build(config)

    def test_context_budget_uses_litellm_context_window(self) -> None:
        with _litellm_patch():
            conv = _factory().build(_make_config())
        assert conv.context_budget.context_window == 128_000

    def test_context_budget_uses_compaction_threshold(self) -> None:
        with _litellm_patch():
            conv = _factory().build(_make_config())
        assert conv.context_budget.compaction_threshold == 0.8

    def test_tool_registered_when_present_in_config(self) -> None:
        config = _make_config(
            tool_registry=[BaseToolConfig(type="stub_tool", name="calc")],
            agent_registry=[_make_agent_config()],
        )
        factory = _factory(tools={"stub_tool": _StubTool})
        with _litellm_patch():
            conv = factory.build(config)
        assert Tool(type="stub_tool", name="calc") in conv._tool_registry.tools  # noqa: SLF001

    def test_multiple_agents_all_registered(self) -> None:
        config = _make_config(
            agent_registry=[_make_agent_config("agent-a"), _make_agent_config("agent-b")],
            default_agent="agent-a",
        )
        with _litellm_patch():
            conv = _factory().build(config)
        assert set(conv._agent_registry.agents) == {"agent-a", "agent-b"}  # noqa: SLF001
