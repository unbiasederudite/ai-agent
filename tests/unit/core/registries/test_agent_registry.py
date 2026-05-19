"""Unit tests for AgentRegistry."""

import pytest

from ai_agent.core.exceptions import AgentNotFoundError
from ai_agent.core.models.agent import Agent
from ai_agent.core.models.llm import LLM, LLMSettings
from ai_agent.core.models.message import Message
from ai_agent.core.models.run import RunResult, RunSettings
from ai_agent.core.models.tool import ToolDefinition
from ai_agent.core.protocols.llm import ILLMProvider
from ai_agent.core.registries.agent import AgentRegistry


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_AGENT_A = Agent(type="node", name="coder")
_AGENT_B = Agent(type="node", name="reviewer")
_LLM = LLM(provider="test", model="test-model")
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=4096)
_RUN_SETTINGS = RunSettings(agent=_AGENT_A, llm=_LLM, settings=_SETTINGS)


class _StubAgent:
    """Minimal IAgent stand-in."""

    def __init__(self, label: str = "a") -> None:
        self.label = label

    @property
    def run_settings(self) -> RunSettings:
        return _RUN_SETTINGS

    def run(
        self,
        messages: list[Message],
        provider: ILLMProvider,
        model: str,
        settings: LLMSettings,
        tools: list[ToolDefinition] | None,
    ) -> RunResult:
        raise NotImplementedError


def _make_registry(*pairs: tuple[Agent, _StubAgent]) -> AgentRegistry:
    registry = AgentRegistry()
    for agent, impl in pairs:
        registry.register(agent, impl)  # type: ignore[arg-type]
    return registry


# ---------------------------------------------------------------------------
# AgentRegistry — register
# ---------------------------------------------------------------------------


class TestAgentRegistryRegister:
    def test_register_adds_agent(self) -> None:
        registry = _make_registry((_AGENT_A, _StubAgent()))
        assert _AGENT_A in registry.agents

    def test_first_registration_wins(self) -> None:
        impl1 = _StubAgent("first")
        impl2 = _StubAgent("second")
        registry = _make_registry((_AGENT_A, impl1), (_AGENT_A, impl2))
        assert registry.resolve_implementation(_AGENT_A).label == "first"  # type: ignore[attr-defined]

    def test_multiple_agents_registered_independently(self) -> None:
        registry = _make_registry((_AGENT_A, _StubAgent("a")), (_AGENT_B, _StubAgent("b")))
        assert set(a.name for a in registry.agents) == {"coder", "reviewer"}


# ---------------------------------------------------------------------------
# AgentRegistry — resolve
# ---------------------------------------------------------------------------


class TestAgentRegistryResolve:
    def test_returns_registered_impl(self) -> None:
        impl = _StubAgent()
        registry = _make_registry((_AGENT_A, impl))
        assert registry.resolve_implementation(_AGENT_A) is impl

    def test_unknown_agent_raises_agent_not_found(self) -> None:
        registry = AgentRegistry()
        with pytest.raises(AgentNotFoundError):
            registry.resolve_implementation(_AGENT_A)

    def test_resolve_correct_impl_among_multiple(self) -> None:
        impl_a = _StubAgent("a")
        impl_b = _StubAgent("b")
        registry = _make_registry((_AGENT_A, impl_a), (_AGENT_B, impl_b))
        assert registry.resolve_implementation(_AGENT_A) is impl_a
        assert registry.resolve_implementation(_AGENT_B) is impl_b


# ---------------------------------------------------------------------------
# AgentRegistry — run_settings access
# ---------------------------------------------------------------------------


class TestAgentRegistryRunSettings:
    def test_resolve_exposes_run_settings(self) -> None:
        registry = _make_registry((_AGENT_A, _StubAgent()))
        assert registry.resolve_implementation(_AGENT_A).run_settings is _RUN_SETTINGS


# ---------------------------------------------------------------------------
# AgentRegistry — agents property
# ---------------------------------------------------------------------------


class TestAgentRegistryAgentsProperty:
    def test_empty_when_no_registrations(self) -> None:
        assert AgentRegistry().agents == []

    def test_returns_registered_agents(self) -> None:
        registry = _make_registry((_AGENT_A, _StubAgent()), (_AGENT_B, _StubAgent()))
        assert set(a.name for a in registry.agents) == {"coder", "reviewer"}

    def test_preserves_registration_order(self) -> None:
        registry = _make_registry((_AGENT_A, _StubAgent()), (_AGENT_B, _StubAgent()))
        assert [a.name for a in registry.agents] == ["coder", "reviewer"]
