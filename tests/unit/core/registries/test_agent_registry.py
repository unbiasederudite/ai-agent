"""Unit tests for AgentRegistry."""

import pytest

from ai_agent.core.exceptions import AgentNotFoundError
from ai_agent.core.models.agent import Agent
from ai_agent.core.models.llm import LLMSettings
from ai_agent.core.models.run import RunSettings
from ai_agent.core.models.llm import LLM
from ai_agent.core.registries.agent import AgentRegistry


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_AGENT_A = Agent(name="coder")
_AGENT_B = Agent(name="reviewer")
_LLM = LLM(provider="test", model="test-model")
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=4096)
_RUN_SETTINGS = RunSettings(llm=_LLM, settings=_SETTINGS)


class _StubNode:
    """Minimal stand-in for AgentNode."""

    def __init__(self, label: str = "a") -> None:
        self.label = label
        self._run_settings = _RUN_SETTINGS

    @property
    def run_settings(self) -> RunSettings:
        return self._run_settings


def _make_registry(*pairs: tuple[Agent, _StubNode]) -> AgentRegistry:
    registry = AgentRegistry()
    for agent, node in pairs:
        registry.register(agent, node)  # type: ignore[arg-type]
    return registry


# ---------------------------------------------------------------------------
# AgentRegistry — register
# ---------------------------------------------------------------------------


class TestAgentRegistryRegister:
    def test_register_adds_agent(self) -> None:
        registry = _make_registry((_AGENT_A, _StubNode()))
        assert "coder" in registry.agents

    def test_first_registration_wins(self) -> None:
        node1 = _StubNode("first")
        node2 = _StubNode("second")
        registry = _make_registry((_AGENT_A, node1), (_AGENT_A, node2))
        assert registry.resolve(_AGENT_A).label == "first"  # type: ignore[attr-defined]

    def test_multiple_agents_registered_independently(self) -> None:
        registry = _make_registry((_AGENT_A, _StubNode("a")), (_AGENT_B, _StubNode("b")))
        assert set(registry.agents) == {"coder", "reviewer"}


# ---------------------------------------------------------------------------
# AgentRegistry — resolve
# ---------------------------------------------------------------------------


class TestAgentRegistryResolve:
    def test_returns_registered_node(self) -> None:
        node = _StubNode()
        registry = _make_registry((_AGENT_A, node))
        assert registry.resolve(_AGENT_A) is node

    def test_unknown_agent_raises_agent_not_found(self) -> None:
        registry = AgentRegistry()
        with pytest.raises(AgentNotFoundError):
            registry.resolve(_AGENT_A)

    def test_resolve_correct_node_among_multiple(self) -> None:
        node_a = _StubNode("a")
        node_b = _StubNode("b")
        registry = _make_registry((_AGENT_A, node_a), (_AGENT_B, node_b))
        assert registry.resolve(_AGENT_A) is node_a
        assert registry.resolve(_AGENT_B) is node_b


# ---------------------------------------------------------------------------
# AgentRegistry — run_settings access
# ---------------------------------------------------------------------------


class TestAgentRegistryRunSettings:
    def test_resolve_exposes_run_settings(self) -> None:
        registry = _make_registry((_AGENT_A, _StubNode()))
        assert registry.resolve(_AGENT_A).run_settings is _RUN_SETTINGS


# ---------------------------------------------------------------------------
# AgentRegistry — agents property
# ---------------------------------------------------------------------------


class TestAgentRegistryAgentsProperty:
    def test_empty_when_no_registrations(self) -> None:
        assert AgentRegistry().agents == []

    def test_returns_registered_names(self) -> None:
        registry = _make_registry((_AGENT_A, _StubNode()), (_AGENT_B, _StubNode()))
        assert set(registry.agents) == {"coder", "reviewer"}

    def test_preserves_registration_order(self) -> None:
        registry = _make_registry((_AGENT_A, _StubNode()), (_AGENT_B, _StubNode()))
        assert registry.agents == ["coder", "reviewer"]
