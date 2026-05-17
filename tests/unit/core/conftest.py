"""Shared fixtures for unit/core tests."""

import pytest

from ai_agent.core.models.config import (
    AgentConfig,
    AgentRegistryConfig,
    ConversationConfig,
    LLMConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
)
from ai_agent.core.models.llm import LLM, LLMSettings
from ai_agent.core.models.strategy import StrategyConfig


@pytest.fixture()
def llm_config() -> LLM:
    """Minimal LLM identity for tests."""
    return LLM(provider="test_provider", model="test_model")


@pytest.fixture()
def agent_config(llm_config: LLM) -> AgentConfig:
    """Minimal AgentConfig for tests."""
    return AgentConfig(
        name="default",
        llm=llm_config,
        strategy=StrategyConfig(type="default"),
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture()
def conversation_config(llm_config: LLM, agent_config: AgentConfig) -> ConversationConfig:
    """Minimal ConversationConfig for tests."""
    return ConversationConfig(
        llm_registry=LLMRegistryConfig(
            registry=[
                LLMProviderConfig(
                    provider=llm_config.provider,
                    models=[
                        LLMConfig(
                            model=llm_config.model,
                            settings=LLMSettings(temperature=0.7, max_tokens=4096),
                        )
                    ],
                )
            ]
        ),
        agent_registry=AgentRegistryConfig(agents=[agent_config]),
    )
