"""Shared fixtures for unit/core tests."""

import pytest

from ai_agent.core.models.agent import Agent, AgentConfig
from ai_agent.core.models.config import (
    AgentRegistryConfig,
    CompactionConfig,
    ConversationConfig,
    LLMConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
)
from ai_agent.core.models.llm import LLM, LLMSettings


@pytest.fixture()
def llm_config() -> LLM:
    """Minimal LLM identity for tests."""
    return LLM(provider="test_provider", model="test_model")


@pytest.fixture()
def agent_config() -> AgentConfig:
    """Minimal AgentConfig for tests."""
    return AgentConfig(type="node", name="default")


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
        default_agent=Agent(type="node", name=agent_config.name),
        compaction=CompactionConfig(
            llm=llm_config,
            temperature=0.3,
            threshold=0.8,
        ),
    )
