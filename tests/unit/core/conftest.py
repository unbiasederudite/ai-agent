"""Shared fixtures for unit/core tests."""

import pytest

from ai_agent.core.models.agent import AgentConfig
from ai_agent.core.models.config import (
    CompactionConfig,
    ConversationConfig,
    LLMConfig,
    LLMProviderConfig,
)
from ai_agent.core.models.llm import LLM, LLMSettings
from ai_agent.core.models.strategy import ReActStrategyConfig


@pytest.fixture()
def llm_config() -> LLM:
    """Minimal LLM identity for tests."""
    return LLM(provider="test_provider", model="test_model")


@pytest.fixture()
def agent_config(llm_config: LLM) -> AgentConfig:
    """Minimal AgentConfig for tests."""
    return AgentConfig(
        name="default",
        description="Default test agent.",
        llm=llm_config,
        settings=LLMSettings(temperature=0.7, max_tokens=4096),
        strategy=ReActStrategyConfig(),
        tools=[],
    )


@pytest.fixture()
def conversation_config(llm_config: LLM, agent_config: AgentConfig) -> ConversationConfig:
    """Minimal ConversationConfig for tests."""
    return ConversationConfig(
        llm_registry=[
            LLMProviderConfig(
                provider=llm_config.provider,
                models=[
                    LLMConfig(
                        model=llm_config.model,
                        settings=LLMSettings(temperature=0.7, max_tokens=4096),
                    )
                ],
            )
        ],
        agent_registry=[agent_config],
        default_agent=agent_config.name,
        compaction=CompactionConfig(
            llm=llm_config,
            settings=LLMSettings(temperature=0.3, max_tokens=2048),
            threshold=0.8,
        ),
    )
