"""Unit tests for AgentFactory."""

from ai_agent.core.factories.agent import AgentFactory
from ai_agent.core.models.agent import AgentConfig, AgentState, StepResult
from ai_agent.core.models.llm import LLM, LLMRequest, LLMSettings
from ai_agent.core.models.strategy import StrategyConfig
from ai_agent.core.models.tool import Tool
from ai_agent.core.protocols.llm import ILLMProvider
from ai_agent.core.registries.tool import ToolRegistry
from ai_agent.core.services.agent import Agent
from ai_agent.core.services.tool import ToolService
from ai_agent.core.strategies.base import BaseStrategy


_LLM = LLM(provider="test", model="test-model")
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=4096)
_STRATEGY_CFG = StrategyConfig(type="stub")
_TOOL = Tool(type="stub", name="calc")


class _StubStrategy(BaseStrategy):
    def step(self, state: AgentState, request: LLMRequest, provider: ILLMProvider) -> StepResult:
        raise NotImplementedError


class _StubStrategyFactory:
    """Always returns a _StubStrategy regardless of config."""

    def build(self, config: StrategyConfig) -> _StubStrategy:
        return _StubStrategy(config, ToolService(registry=ToolRegistry()))


def _make_config(**overrides: object) -> AgentConfig:
    defaults: dict[str, object] = {
        "name": "coder",
        "description": "A coding agent.",
        "llm": _LLM,
        "settings": _SETTINGS,
        "strategy": _STRATEGY_CFG,
        "tools": [],
    }
    return AgentConfig(**{**defaults, **overrides})  # type: ignore[arg-type]


def _factory() -> AgentFactory:
    return AgentFactory(strategy_factory=_StubStrategyFactory())  # type: ignore[arg-type]


class TestAgentFactoryBuild:
    def test_returns_agent_instance(self) -> None:
        agent = _factory().build(_make_config())
        assert isinstance(agent, Agent)

    def test_agent_name_matches_config(self) -> None:
        agent = _factory().build(_make_config(name="reviewer"))
        assert agent.name == "reviewer"

    def test_agent_description_matches_config(self) -> None:
        agent = _factory().build(_make_config(description="Reviews code."))
        assert agent.description == "Reviews code."

    def test_run_settings_llm_matches_config(self) -> None:
        agent = _factory().build(_make_config())
        assert agent.run_settings.llm == _LLM

    def test_run_settings_settings_matches_config(self) -> None:
        agent = _factory().build(_make_config())
        assert agent.run_settings.settings == _SETTINGS

    def test_run_settings_agent_name_matches_config(self) -> None:
        agent = _factory().build(_make_config(name="coder"))
        assert agent.run_settings.agent == "coder"

    def test_run_settings_tools_matches_config(self) -> None:
        agent = _factory().build(_make_config(tools=[_TOOL]))
        assert agent.run_settings.tools == [_TOOL]

    def test_each_call_returns_new_instance(self) -> None:
        factory = _factory()
        config = _make_config()
        assert factory.build(config) is not factory.build(config)
