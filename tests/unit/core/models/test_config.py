"""Unit tests for config models."""

import pytest
from pydantic import ValidationError

from ai_agent.core.models.config import (
    AgentConfig,
    AgentRegistryConfig,
    CompactionConfig,
    ConversationConfig,
    LLMConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
    LoggingConfig,
    ToolRegistryConfig,
)
from ai_agent.core.models.llm import LLM, LLMSettings
from ai_agent.core.models.strategy import StrategyConfig
from ai_agent.core.models.tool import Tool, ToolConfig


class TestLLMConfig:
    """Tests for the LLMConfig model (model name + settings)."""

    def _make(self, **overrides: object) -> LLMConfig:
        defaults: dict[str, object] = {
            "model": "gpt-4o",
            "settings": LLMSettings(temperature=0.7, max_tokens=4096),
        }
        return LLMConfig(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_llm_config_constructs(self) -> None:
        cfg = self._make()
        assert cfg.model == "gpt-4o"
        assert cfg.settings.temperature == 0.7
        assert cfg.settings.max_tokens == 4096

    def test_llm_config_is_frozen(self) -> None:
        cfg = self._make()
        with pytest.raises(Exception):
            cfg.model = "other"  # type: ignore[misc]

    def test_llm_config_requires_model(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            LLMConfig(settings=LLMSettings(temperature=0.7, max_tokens=4096))  # type: ignore[call-arg]

    def test_llm_config_requires_settings(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            LLMConfig(model="gpt-4o")  # type: ignore[call-arg]

    def test_llm_config_settings_field_is_llm_settings(self) -> None:
        cfg = self._make()
        assert isinstance(cfg.settings, LLMSettings)

    def test_llm_config_rejects_invalid_temperature(self) -> None:
        with pytest.raises(ValidationError):
            self._make(settings=LLMSettings(temperature=2.1, max_tokens=4096))

    def test_llm_config_rejects_zero_max_tokens(self) -> None:
        with pytest.raises(ValidationError):
            self._make(settings=LLMSettings(temperature=0.7, max_tokens=0))

    def test_llm_config_has_no_flat_temperature_field(self) -> None:
        assert "temperature" not in LLMConfig.model_fields

    def test_llm_config_has_no_flat_max_tokens_field(self) -> None:
        assert "max_tokens" not in LLMConfig.model_fields

    def test_llm_config_has_no_llm_field(self) -> None:
        assert "llm" not in LLMConfig.model_fields


class TestStrategyConfig:
    """Tests for the StrategyConfig model."""

    def test_strategy_config_defaults(self) -> None:
        cfg = StrategyConfig(type="default")
        assert cfg.max_turns >= 1

    def test_strategy_config_is_frozen(self) -> None:
        cfg = StrategyConfig(type="default")
        with pytest.raises(Exception):
            cfg.max_turns = 99  # type: ignore[misc]

    def test_strategy_config_rejects_zero_turns(self) -> None:
        with pytest.raises(ValidationError):
            StrategyConfig(type="default", max_turns=0)

    def test_strategy_config_has_no_max_tokens_field(self) -> None:
        assert "max_tokens" not in StrategyConfig.model_fields


class TestCompactionConfig:
    """Tests for CompactionConfig — LLM-based session compaction settings."""

    def test_compaction_config_defaults(self) -> None:
        cfg = CompactionConfig()
        assert cfg.max_tokens >= 1

    def test_compaction_config_is_frozen(self) -> None:
        cfg = CompactionConfig()
        with pytest.raises(Exception):
            cfg.max_tokens = 9999  # type: ignore[misc]

    def test_compaction_config_rejects_zero_max_tokens(self) -> None:
        with pytest.raises(ValidationError):
            CompactionConfig(max_tokens=0)

    def test_compaction_config_max_tokens_can_be_set(self) -> None:
        cfg = CompactionConfig(max_tokens=4096)
        assert cfg.max_tokens == 4096

    def test_compaction_config_has_no_ratio_field(self) -> None:
        assert "ratio" not in CompactionConfig.model_fields

    def test_conversation_config_compaction_defaults(
        self, conversation_config: ConversationConfig
    ) -> None:
        assert isinstance(conversation_config.compaction, CompactionConfig)
        assert conversation_config.compaction.max_tokens >= 1


class TestToolRegistryConfig:
    """Tests for the ToolRegistryConfig model."""

    def test_tool_registry_config_defaults_empty(self) -> None:
        cfg = ToolRegistryConfig()
        assert cfg.tools == []

    def test_tool_registry_config_stores_tool_configs(self) -> None:
        tools = [ToolConfig(type="test", name="calculator"), ToolConfig(type="test", name="search")]
        cfg = ToolRegistryConfig(tools=tools)
        assert any(t.name == "calculator" for t in cfg.tools)


class TestLoggingConfig:
    """Tests for the LoggingConfig model."""

    def test_logging_config_defaults(self) -> None:
        cfg = LoggingConfig()
        assert cfg.level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    def test_logging_config_rejects_invalid_level(self) -> None:
        with pytest.raises(ValidationError):
            LoggingConfig(level="VERBOSE")  # type: ignore[arg-type]


class TestAgentConfig:
    """Tests for the per-agent AgentConfig model."""

    def _make(self, **overrides: object) -> AgentConfig:
        defaults: dict[str, object] = {
            "name": "default",
            "llm": LLM(provider="test", model="test-model"),
            "strategy": StrategyConfig(type="default"),
            "system_prompt": "You are a helpful assistant.",
        }
        return AgentConfig(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_agent_config_constructs_with_required_fields(self) -> None:
        cfg = self._make()
        assert cfg.name == "default"
        assert cfg.llm.provider == "test"
        assert cfg.llm.model == "test-model"
        assert cfg.strategy is not None
        assert cfg.system_prompt == "You are a helpful assistant."
        assert cfg.tools == []

    def test_agent_config_requires_llm(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            AgentConfig(name="x", strategy=StrategyConfig(type="default"), system_prompt="x")  # type: ignore[call-arg]

    def test_agent_config_is_frozen(self) -> None:
        cfg = self._make()
        with pytest.raises(Exception):
            cfg.strategy = StrategyConfig(type="other")  # type: ignore[misc]

    def test_agent_config_strategy_accepts_explicit_values(self) -> None:
        cfg = self._make(strategy=StrategyConfig(type="default", max_turns=5))
        assert isinstance(cfg.strategy, StrategyConfig)
        assert cfg.strategy.max_turns == 5

    def test_agent_config_requires_system_prompt(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            AgentConfig(
                name="x",
                llm=LLM(provider="test", model="m"),
                strategy=StrategyConfig(type="default"),
            )  # type: ignore[call-arg]

    def test_agent_config_system_prompt_can_be_set(self) -> None:
        cfg = self._make(system_prompt="You are a pirate.")
        assert cfg.system_prompt == "You are a pirate."

    def test_agent_config_has_no_llm_registry_field(self) -> None:
        cfg = self._make()
        assert not hasattr(cfg, "llm_registry")

    def test_agent_config_has_no_tool_registry_field(self) -> None:
        cfg = self._make()
        assert not hasattr(cfg, "tool_registry")

    def test_agent_config_tools_defaults_empty(self) -> None:
        cfg = self._make()
        assert cfg.tools == []

    def test_agent_config_tools_can_be_set(self) -> None:
        tools = [Tool(type="mcp", name="search"), Tool(type="mcp", name="calc")]
        cfg = self._make(tools=tools)
        assert len(cfg.tools) == 2
        assert cfg.tools[0].name == "search"

    def test_agent_config_tools_field_contains_tool_instances(self) -> None:
        tools = [Tool(type="mcp", name="search")]
        cfg = self._make(tools=tools)
        assert all(isinstance(t, Tool) for t in cfg.tools)


class TestConversationConfig:
    """Tests for the global ConversationConfig model."""

    def _make_provider_cfg(self) -> LLMProviderConfig:
        return LLMProviderConfig(
            provider="test",
            models=[
                LLMConfig(
                    model="test-model", settings=LLMSettings(temperature=0.7, max_tokens=4096)
                )
            ],
        )

    def _make(self, **overrides: object) -> ConversationConfig:
        agent = AgentConfig(
            name="default",
            llm=LLM(provider="test", model="test-model"),
            strategy=StrategyConfig(type="default"),
            system_prompt="You are a helpful assistant.",
        )
        defaults: dict[str, object] = {
            "llm_registry": LLMRegistryConfig(registry=[self._make_provider_cfg()]),
            "agent_registry": AgentRegistryConfig(agents=[agent]),
        }
        return ConversationConfig(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_conversation_config_constructs_with_required_fields(self) -> None:
        cfg = self._make()
        assert cfg.llm_registry is not None
        assert cfg.agent_registry is not None
        assert cfg.tool_registry is None
        assert cfg.compaction is not None
        assert cfg.logging is not None

    def test_conversation_config_requires_llm_registry(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            agent = AgentConfig(
                name="x",
                llm=LLM(provider="test", model="test-model"),
                strategy=StrategyConfig(type="default"),
                system_prompt="x",
            )
            ConversationConfig(agent_registry=AgentRegistryConfig(agents=[agent]))  # type: ignore[call-arg]

    def test_conversation_config_requires_agent_registry(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            ConversationConfig(  # type: ignore[call-arg]
                llm_registry=LLMRegistryConfig(registry=[self._make_provider_cfg()])
            )

    def test_conversation_config_is_frozen(self) -> None:
        cfg = self._make()
        with pytest.raises(Exception):
            cfg.logging = LoggingConfig()  # type: ignore[misc]

    def test_conversation_config_tool_registry_defaults_to_none(self) -> None:
        cfg = self._make()
        assert cfg.tool_registry is None

    def test_conversation_config_tool_registry_can_be_set(self) -> None:
        tool = ToolConfig(type="test", name="calc")
        cfg = self._make(tool_registry=ToolRegistryConfig(tools=[tool]))
        assert cfg.tool_registry is not None
        assert cfg.tool_registry.tools[0].name == "calc"

    def test_conversation_config_llm_registry_preserved(self) -> None:
        provider_cfg = LLMProviderConfig(
            provider="test",
            models=[
                LLMConfig(model="a", settings=LLMSettings(temperature=0.7, max_tokens=4096)),
                LLMConfig(model="b", settings=LLMSettings(temperature=0.5, max_tokens=2048)),
            ],
        )
        cfg = self._make(llm_registry=LLMRegistryConfig(registry=[provider_cfg]))
        assert len(cfg.llm_registry.registry) == 1
        assert len(cfg.llm_registry.registry[0].models) == 2

    def test_conversation_config_agent_registry_preserved(self) -> None:
        agents = [
            AgentConfig(
                name="agent-a",
                llm=LLM(provider="test", model="a"),
                strategy=StrategyConfig(type="default"),
                system_prompt="System A.",
            ),
            AgentConfig(
                name="agent-b",
                llm=LLM(provider="test", model="b"),
                strategy=StrategyConfig(type="default"),
                system_prompt="System B.",
            ),
        ]
        cfg = self._make(agent_registry=AgentRegistryConfig(agents=agents))
        assert len(cfg.agent_registry.agents) == 2
