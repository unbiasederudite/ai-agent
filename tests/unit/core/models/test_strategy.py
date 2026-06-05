"""Unit tests for Strategy, BaseStrategyConfig, ReActStrategyConfig, and StrategyConfig."""

import pytest
from pydantic import ValidationError

from ai_agent.core.models.strategy import (
    BaseStrategyConfig,
    ReActStrategyConfig,
    Strategy,
    StrategyConfig,
)


class TestStrategy:
    def test_constructs_with_type(self) -> None:
        s = Strategy(type="foo")
        assert s.type == "foo"

    def test_is_frozen(self) -> None:
        s = Strategy(type="foo")
        with pytest.raises(Exception):
            s.type = "bar"  # type: ignore[misc]

    def test_has_type_field(self) -> None:
        assert "type" in Strategy.model_fields


class TestBaseStrategyConfig:
    def test_max_turns_defaults_to_ten(self) -> None:
        cfg = BaseStrategyConfig(type="default")
        assert cfg.max_turns == 10

    def test_is_frozen(self) -> None:
        cfg = BaseStrategyConfig(type="default")
        with pytest.raises(Exception):
            cfg.max_turns = 99  # type: ignore[misc]

    def test_rejects_zero_turns(self) -> None:
        with pytest.raises(ValidationError):
            BaseStrategyConfig(type="default", max_turns=0)

    def test_has_no_max_tokens_field(self) -> None:
        assert "max_tokens" not in BaseStrategyConfig.model_fields


class TestReActStrategyConfig:
    def test_default_type_is_react(self) -> None:
        assert ReActStrategyConfig().type == "react"

    def test_inherits_max_turns_default(self) -> None:
        assert ReActStrategyConfig().max_turns == 10

    def test_custom_max_turns(self) -> None:
        assert ReActStrategyConfig(max_turns=5).max_turns == 5

    def test_is_base_strategy_config(self) -> None:
        assert isinstance(ReActStrategyConfig(), BaseStrategyConfig)


class TestStrategyConfig:
    def test_is_react_strategy_config(self) -> None:
        assert StrategyConfig is ReActStrategyConfig
