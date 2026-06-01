"""Unit tests for StrategyFactory."""

import pytest

from ai_agent.core.exceptions import ConfigError
from ai_agent.core.factories.strategy import StrategyFactory
from ai_agent.core.models.agent import AgentState, StepResult
from ai_agent.core.models.llm import LLMRequest
from ai_agent.core.models.strategy import StrategyConfig
from ai_agent.core.protocols.llm import ILLMProvider
from ai_agent.core.registries.tool import ToolRegistry
from ai_agent.core.services.tool import ToolService
from ai_agent.core.strategies.base import BaseStrategy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CONFIG_A = StrategyConfig(type="alpha", max_turns=5)
_CONFIG_B = StrategyConfig(type="beta", max_turns=10)


def _tool_service() -> ToolService:
    return ToolService(registry=ToolRegistry())


class _AlphaStrategy(BaseStrategy):
    def step(self, state: AgentState, request: LLMRequest, provider: ILLMProvider) -> StepResult:
        raise NotImplementedError


class _BetaStrategy(BaseStrategy):
    def step(self, state: AgentState, request: LLMRequest, provider: ILLMProvider) -> StepResult:
        raise NotImplementedError


def _factory(implementations: dict[str, type[BaseStrategy]]) -> StrategyFactory:
    return StrategyFactory(
        implementations=implementations,
        tool_service=_tool_service(),
    )


# ---------------------------------------------------------------------------
# StrategyFactory — build
# ---------------------------------------------------------------------------


class TestStrategyFactoryBuild:
    def test_returns_instance_of_registered_class(self) -> None:
        factory = _factory({"alpha": _AlphaStrategy})
        assert isinstance(factory.build(_CONFIG_A), _AlphaStrategy)

    def test_strategy_receives_config(self) -> None:
        factory = _factory({"alpha": _AlphaStrategy})
        assert factory.build(_CONFIG_A).config is _CONFIG_A

    def test_selects_correct_class_among_multiple(self) -> None:
        factory = _factory({"alpha": _AlphaStrategy, "beta": _BetaStrategy})
        assert isinstance(factory.build(_CONFIG_A), _AlphaStrategy)
        assert isinstance(factory.build(_CONFIG_B), _BetaStrategy)

    def test_each_call_returns_new_instance(self) -> None:
        factory = _factory({"alpha": _AlphaStrategy})
        assert factory.build(_CONFIG_A) is not factory.build(_CONFIG_A)

    def test_unknown_type_raises_config_error(self) -> None:
        factory = _factory({})
        with pytest.raises(ConfigError):
            factory.build(_CONFIG_A)

    def test_error_message_contains_unknown_type(self) -> None:
        factory = _factory({})
        with pytest.raises(ConfigError, match="alpha"):
            factory.build(_CONFIG_A)

    def test_error_message_lists_available_types(self) -> None:
        factory = _factory({"beta": _BetaStrategy})
        with pytest.raises(ConfigError, match="beta"):
            factory.build(_CONFIG_A)


# ---------------------------------------------------------------------------
# StrategyFactory — independent instances
# ---------------------------------------------------------------------------


class TestStrategyFactoryInit:
    def test_independent_instances_have_independent_implementations(self) -> None:
        fa = _factory({"alpha": _AlphaStrategy})
        fb = _factory({"beta": _BetaStrategy})

        assert isinstance(fa.build(_CONFIG_A), _AlphaStrategy)
        assert isinstance(fb.build(_CONFIG_B), _BetaStrategy)

        with pytest.raises(ConfigError):
            fa.build(_CONFIG_B)
        with pytest.raises(ConfigError):
            fb.build(_CONFIG_A)
