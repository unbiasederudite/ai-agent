"""Unit tests for ToolFactory."""

import pytest

from ai_agent.core.exceptions import ConfigError
from ai_agent.core.factories.tool import ToolFactory
from ai_agent.core.models.tool import BaseToolConfig, ToolResponse, ToolSchema
from ai_agent.core.protocols.tool import ITool
from ai_agent.core.tools.base import BaseTool


_CONFIG_A = BaseToolConfig(type="alpha", name="tool-a")
_CONFIG_B = BaseToolConfig(type="beta", name="tool-b")

_SCHEMA = ToolSchema(description="stub", parameters={"type": "object", "properties": {}})


class _AlphaTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return _SCHEMA

    def execute(self, arguments: dict[str, object]) -> ToolResponse:
        return ToolResponse(content="alpha")


class _BetaTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return _SCHEMA

    def execute(self, arguments: dict[str, object]) -> ToolResponse:
        return ToolResponse(content="beta")


def _factory(implementations: dict[str, type[BaseTool]]) -> ToolFactory:
    return ToolFactory(implementations=implementations)


class TestToolFactoryBuild:
    def test_returns_instance_of_registered_class(self) -> None:
        factory = _factory({"alpha": _AlphaTool})
        assert isinstance(factory.build(_CONFIG_A), _AlphaTool)

    def test_returns_itool(self) -> None:
        factory = _factory({"alpha": _AlphaTool})
        assert isinstance(factory.build(_CONFIG_A), ITool)

    def test_selects_correct_class_among_multiple(self) -> None:
        factory = _factory({"alpha": _AlphaTool, "beta": _BetaTool})
        assert isinstance(factory.build(_CONFIG_A), _AlphaTool)
        assert isinstance(factory.build(_CONFIG_B), _BetaTool)

    def test_each_call_returns_new_instance(self) -> None:
        factory = _factory({"alpha": _AlphaTool})
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
        factory = _factory({"beta": _BetaTool})
        with pytest.raises(ConfigError, match="beta"):
            factory.build(_CONFIG_A)
