"""Unit tests for config models."""

import pytest
from pydantic import ValidationError

from ai_agent.core.models.agent import Agent, AgentConfig
from ai_agent.core.models.config import (
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
from ai_agent.core.models.tool import ToolConfig


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

    def _make(self, **overrides: object) -> CompactionConfig:
        defaults: dict[str, object] = {
            "llm": LLM(provider="test", model="test-model"),
            "temperature": 0.3,
            "threshold": 0.8,
        }
        return CompactionConfig(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_compaction_config_constructs(self) -> None:
        cfg = self._make()
        assert cfg.max_tokens >= 1

    def test_compaction_config_is_frozen(self) -> None:
        cfg = self._make()
        with pytest.raises(Exception):
            cfg.max_tokens = 9999  # type: ignore[misc]

    def test_compaction_config_rejects_zero_max_tokens(self) -> None:
        with pytest.raises(ValidationError):
            self._make(max_tokens=0)

    def test_compaction_config_max_tokens_can_be_set(self) -> None:
        cfg = self._make(max_tokens=4096)
        assert cfg.max_tokens == 4096

    def test_compaction_config_has_no_ratio_field(self) -> None:
        assert "ratio" not in CompactionConfig.model_fields

    def test_compaction_config_llm_stored(self) -> None:
        cfg = self._make(llm=LLM(provider="p", model="m"))
        assert cfg.llm.provider == "p"
        assert cfg.llm.model == "m"

    def test_compaction_config_temperature_stored(self) -> None:
        cfg = self._make(temperature=0.5)
        assert cfg.temperature == 0.5

    def test_compaction_config_threshold_stored(self) -> None:
        cfg = self._make(threshold=0.75)
        assert cfg.threshold == 0.75

    def test_compaction_config_rejects_threshold_above_one(self) -> None:
        with pytest.raises(ValidationError):
            self._make(threshold=1.1)

    def test_compaction_config_rejects_threshold_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            self._make(threshold=-0.1)

    def test_compaction_config_rejects_invalid_temperature(self) -> None:
        with pytest.raises(ValidationError):
            self._make(temperature=2.1)

    def test_conversation_config_compaction_stored(
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
    """Tests for the AgentConfig base model."""

    def test_agent_config_constructs_with_type_and_name(self) -> None:
        cfg = AgentConfig(type="node", name="default")
        assert cfg.type == "node"
        assert cfg.name == "default"

    def test_agent_config_requires_type(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            AgentConfig(name="x")  # type: ignore[call-arg]

    def test_agent_config_requires_name(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            AgentConfig(type="node")  # type: ignore[call-arg]

    def test_agent_config_is_frozen(self) -> None:
        cfg = AgentConfig(type="node", name="default")
        with pytest.raises(Exception):
            cfg.name = "other"  # type: ignore[misc]

    def test_agent_config_has_no_llm_field(self) -> None:
        assert "llm" not in AgentConfig.model_fields

    def test_agent_config_has_no_system_prompt_field(self) -> None:
        assert "system_prompt" not in AgentConfig.model_fields

    def test_agent_config_has_no_strategy_field(self) -> None:
        assert "strategy" not in AgentConfig.model_fields


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
        defaults: dict[str, object] = {
            "llm_registry": LLMRegistryConfig(registry=[self._make_provider_cfg()]),
            "agent_registry": AgentRegistryConfig(
                agents=[AgentConfig(type="node", name="default")]
            ),
            "default_agent": Agent(type="node", name="default"),
            "compaction": CompactionConfig(
                llm=LLM(provider="test", model="test-model"),
                temperature=0.3,
                threshold=0.8,
            ),
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
            ConversationConfig(
                agent_registry=AgentRegistryConfig(agents=[AgentConfig(type="node", name="x")])
            )  # type: ignore[call-arg]

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
        cfg = self._make(
            llm_registry=LLMRegistryConfig(registry=[provider_cfg]),
            compaction=CompactionConfig(
                llm=LLM(provider="test", model="a"),
                temperature=0.3,
                threshold=0.8,
            ),
        )
        assert len(cfg.llm_registry.registry) == 1
        assert len(cfg.llm_registry.registry[0].models) == 2

    def test_conversation_config_agent_registry_preserved(self) -> None:
        agents = [
            AgentConfig(type="node", name="agent-a"),
            AgentConfig(type="node", name="agent-b"),
        ]
        cfg = self._make(
            agent_registry=AgentRegistryConfig(agents=agents),
            default_agent=Agent(type="node", name="agent-a"),
        )
        assert len(cfg.agent_registry.agents) == 2

    def test_default_agent_preserved(self) -> None:
        cfg = self._make()
        assert cfg.default_agent.name == "default"

    def test_compaction_llm_preserved(self) -> None:
        cfg = self._make()
        assert cfg.compaction.llm.provider == "test"
        assert cfg.compaction.llm.model == "test-model"

    def test_default_agent_not_in_registry_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._make(default_agent=Agent(type="node", name="nonexistent"))

    def test_compaction_llm_not_in_registry_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._make(
                compaction=CompactionConfig(
                    llm=LLM(provider="other", model="other-model"),
                    temperature=0.3,
                    threshold=0.8,
                )
            )

    def test_message_char_limit_defaults(self) -> None:
        cfg = self._make()
        assert cfg.message_char_limit >= 1

    def test_message_char_limit_can_be_set(self) -> None:
        cfg = self._make(message_char_limit=500)
        assert cfg.message_char_limit == 500

    def test_message_char_limit_rejects_zero(self) -> None:
        with pytest.raises(ValidationError):
            self._make(message_char_limit=0)

    def test_message_char_limit_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            self._make(message_char_limit=-1)
