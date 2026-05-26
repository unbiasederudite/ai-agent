"""Unit tests for config models."""

import pytest
from pydantic import ValidationError

from ai_agent.core.models.agent import AgentConfig
from ai_agent.core.models.config import (
    CompactionConfig,
    ConversationConfig,
    LLMConfig,
    LLMProviderConfig,
    LoggingConfig,
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

    def _make(self, **overrides: object) -> CompactionConfig:
        defaults: dict[str, object] = {
            "llm": LLM(provider="test", model="test-model"),
            "settings": LLMSettings(temperature=0.3, max_tokens=2048),
            "threshold": 0.8,
        }
        return CompactionConfig(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_compaction_config_constructs(self) -> None:
        cfg = self._make()
        assert cfg.settings.max_tokens >= 1

    def test_compaction_config_is_frozen(self) -> None:
        cfg = self._make()
        with pytest.raises(Exception):
            cfg.threshold = 0.5  # type: ignore[misc]

    def test_compaction_config_has_no_ratio_field(self) -> None:
        assert "ratio" not in CompactionConfig.model_fields

    def test_compaction_config_has_no_flat_temperature_field(self) -> None:
        assert "temperature" not in CompactionConfig.model_fields

    def test_compaction_config_has_no_flat_max_tokens_field(self) -> None:
        assert "max_tokens" not in CompactionConfig.model_fields

    def test_compaction_config_llm_stored(self) -> None:
        cfg = self._make(llm=LLM(provider="p", model="m"))
        assert cfg.llm.provider == "p"
        assert cfg.llm.model == "m"

    def test_compaction_config_settings_stored(self) -> None:
        cfg = self._make(settings=LLMSettings(temperature=0.5, max_tokens=1024))
        assert cfg.settings.temperature == 0.5
        assert cfg.settings.max_tokens == 1024

    def test_compaction_config_threshold_stored(self) -> None:
        cfg = self._make(threshold=0.75)
        assert cfg.threshold == 0.75

    def test_compaction_config_rejects_threshold_above_one(self) -> None:
        with pytest.raises(ValidationError):
            self._make(threshold=1.1)

    def test_compaction_config_rejects_threshold_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            self._make(threshold=-0.1)

    def test_compaction_config_rejects_invalid_settings(self) -> None:
        with pytest.raises(ValidationError):
            self._make(settings=LLMSettings(temperature=2.1, max_tokens=2048))

    def test_conversation_config_compaction_stored(
        self, conversation_config: ConversationConfig
    ) -> None:
        assert isinstance(conversation_config.compaction, CompactionConfig)
        assert conversation_config.compaction.settings.max_tokens >= 1


class TestLoggingConfig:
    """Tests for the LoggingConfig model."""

    def test_logging_config_defaults(self) -> None:
        cfg = LoggingConfig()
        assert cfg.level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    def test_logging_config_rejects_invalid_level(self) -> None:
        with pytest.raises(ValidationError):
            LoggingConfig(level="VERBOSE")  # type: ignore[arg-type]


_LLM = LLM(provider="test", model="test-model")
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=4096)
_STRATEGY = StrategyConfig(type="cot")


class TestAgentConfig:
    """Tests for the flattened AgentConfig model."""

    def _make(self, **overrides: object) -> AgentConfig:
        defaults: dict[str, object] = {
            "name": "my-agent",
            "description": "Test agent.",
            "llm": _LLM,
            "settings": _SETTINGS,
            "strategy": _STRATEGY,
        }
        return AgentConfig(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_constructs_with_required_fields(self) -> None:
        cfg = self._make()
        assert cfg.name == "my-agent"
        assert cfg.llm == _LLM
        assert cfg.settings == _SETTINGS
        assert cfg.strategy == _STRATEGY

    def test_description_is_required(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            AgentConfig(name="x", llm=_LLM, settings=_SETTINGS, strategy=_STRATEGY)  # type: ignore[call-arg]

    def test_description_can_be_set(self) -> None:
        cfg = self._make(description="A helpful agent.")
        assert cfg.description == "A helpful agent."

    def test_system_prompt_defaults_to_empty(self) -> None:
        assert self._make().system_prompt == ""

    def test_system_prompt_can_be_set(self) -> None:
        cfg = self._make(system_prompt="You are helpful.")
        assert cfg.system_prompt == "You are helpful."

    def test_tools_defaults_to_empty(self) -> None:
        assert self._make().tools == []

    def test_tools_empty_list_stored(self) -> None:
        assert self._make(tools=[]).tools == []

    def test_tools_subset_stored(self) -> None:
        tool = Tool(type="stub", name="calc")
        cfg = self._make(tools=[tool])
        assert cfg.tools == [tool]

    def test_requires_llm(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            AgentConfig(name="x", settings=_SETTINGS, strategy=_STRATEGY)  # type: ignore[call-arg]

    def test_requires_settings(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            AgentConfig(name="x", llm=_LLM, strategy=_STRATEGY)  # type: ignore[call-arg]

    def test_requires_strategy(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            AgentConfig(name="x", llm=_LLM, settings=_SETTINGS)  # type: ignore[call-arg]

    def test_is_frozen(self) -> None:
        cfg = self._make()
        with pytest.raises(Exception):
            cfg.system_prompt = "new"  # type: ignore[misc]

    def test_has_no_type_field(self) -> None:
        assert "type" not in AgentConfig.model_fields


class TestConversationConfig:
    """Tests for the global ConversationConfig model."""

    def _make_provider_cfg(self, model: str = "test-model") -> LLMProviderConfig:
        return LLMProviderConfig(
            provider="test",
            models=[LLMConfig(model=model, settings=LLMSettings(temperature=0.7, max_tokens=4096))],
        )

    def _make_agent_cfg(self, name: str = "default") -> AgentConfig:
        return AgentConfig(
            name=name,
            description="Test agent.",
            llm=LLM(provider="test", model="test-model"),
            settings=LLMSettings(temperature=0.7, max_tokens=4096),
            strategy=StrategyConfig(type="cot"),
        )

    def _make(self, **overrides: object) -> ConversationConfig:
        defaults: dict[str, object] = {
            "llm_registry": [self._make_provider_cfg()],
            "agent_registry": [self._make_agent_cfg()],
            "default_agent": "default",
            "compaction": CompactionConfig(
                llm=LLM(provider="test", model="test-model"),
                settings=LLMSettings(temperature=0.3, max_tokens=2048),
                threshold=0.8,
            ),
        }
        return ConversationConfig(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_conversation_config_constructs_with_required_fields(self) -> None:
        cfg = self._make()
        assert cfg.llm_registry is not None
        assert cfg.agent_registry is not None
        assert cfg.tool_registry == []
        assert cfg.compaction is not None
        assert cfg.logging is not None

    def test_conversation_config_requires_llm_registry(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            ConversationConfig(  # type: ignore[call-arg]
                agent_registry=[self._make_agent_cfg()]
            )

    def test_conversation_config_requires_agent_registry(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            ConversationConfig(  # type: ignore[call-arg]
                llm_registry=[self._make_provider_cfg()]
            )

    def test_conversation_config_is_frozen(self) -> None:
        cfg = self._make()
        with pytest.raises(Exception):
            cfg.logging = LoggingConfig()  # type: ignore[misc]

    def test_conversation_config_tool_registry_defaults_to_empty(self) -> None:
        cfg = self._make()
        assert cfg.tool_registry == []

    def test_conversation_config_tool_registry_can_be_set(self) -> None:
        tool = ToolConfig(type="test", name="calc")
        cfg = self._make(tool_registry=[tool])
        assert cfg.tool_registry[0].name == "calc"

    def test_conversation_config_llm_registry_preserved(self) -> None:
        provider_cfg = LLMProviderConfig(
            provider="test",
            models=[
                LLMConfig(model="a", settings=LLMSettings(temperature=0.7, max_tokens=4096)),
                LLMConfig(model="b", settings=LLMSettings(temperature=0.5, max_tokens=2048)),
            ],
        )
        cfg = self._make(
            llm_registry=[provider_cfg],
            agent_registry=[
                AgentConfig(
                    name="default",
                    description="Test agent.",
                    llm=LLM(provider="test", model="a"),
                    settings=LLMSettings(temperature=0.7, max_tokens=4096),
                    strategy=StrategyConfig(type="cot"),
                )
            ],
            compaction=CompactionConfig(
                llm=LLM(provider="test", model="a"),
                settings=LLMSettings(temperature=0.3, max_tokens=2048),
                threshold=0.8,
            ),
        )
        assert len(cfg.llm_registry) == 1
        assert len(cfg.llm_registry[0].models) == 2

    def test_conversation_config_agent_registry_preserved(self) -> None:
        cfg = self._make(
            agent_registry=[self._make_agent_cfg("agent-a"), self._make_agent_cfg("agent-b")],
            default_agent="agent-a",
        )
        assert len(cfg.agent_registry) == 2

    def test_default_agent_preserved(self) -> None:
        cfg = self._make()
        assert cfg.default_agent == "default"

    def test_compaction_llm_preserved(self) -> None:
        cfg = self._make()
        assert cfg.compaction.llm.provider == "test"
        assert cfg.compaction.llm.model == "test-model"

    def test_default_agent_not_in_registry_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._make(default_agent="nonexistent")

    def test_compaction_llm_not_in_registry_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._make(
                compaction=CompactionConfig(
                    llm=LLM(provider="other", model="other-model"),
                    settings=LLMSettings(temperature=0.3, max_tokens=2048),
                    threshold=0.8,
                )
            )

    def test_agent_llm_not_in_registry_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._make(
                agent_registry=[
                    AgentConfig(
                        name="default",
                        description="Test agent.",
                        llm=LLM(provider="unknown", model="unknown-model"),
                        settings=LLMSettings(temperature=0.7, max_tokens=4096),
                        strategy=StrategyConfig(type="cot"),
                    )
                ]
            )

    def test_agent_tool_not_registered_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._make(
                tool_registry=[ToolConfig(type="stub", name="known")],
                agent_registry=[
                    AgentConfig(
                        name="default",
                        description="Test agent.",
                        llm=LLM(provider="test", model="test-model"),
                        settings=LLMSettings(temperature=0.7, max_tokens=4096),
                        strategy=StrategyConfig(type="cot"),
                        tools=[Tool(type="stub", name="unknown-tool")],
                    )
                ],
            )

    def test_agent_tools_empty_skips_tool_validation(self) -> None:
        cfg = self._make()
        assert cfg.agent_registry[0].tools == []

    def test_agent_tools_subset_validated_against_registry(self) -> None:
        cfg = self._make(
            tool_registry=[ToolConfig(type="stub", name="calc")],
            agent_registry=[
                AgentConfig(
                    name="default",
                    description="Test agent.",
                    llm=LLM(provider="test", model="test-model"),
                    settings=LLMSettings(temperature=0.7, max_tokens=4096),
                    strategy=StrategyConfig(type="cot"),
                    tools=[Tool(type="stub", name="calc")],
                )
            ],
        )
        assert cfg.agent_registry[0].tools is not None

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
