"""Configuration models for the AI agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_agent.core.models.llm import LLM, LLMSettings
from ai_agent.core.models.strategy import StrategyConfig
from ai_agent.core.models.tool import Tool, ToolConfig


class LLMConfig(BaseModel):
    """Model name and sampling defaults."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(description="Model identifier string.")
    settings: LLMSettings = Field(description="Sampling defaults.")


class LLMProviderConfig(BaseModel):
    """Provider name and its available models."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(description="Provider name.")
    models: list[LLMConfig] = Field(
        min_length=1,
        description="Model configurations for this provider.",
    )


class ToolRegistryConfig(BaseModel):
    """Tool configurations exposed to the LLM."""

    model_config = ConfigDict(frozen=True)

    tools: list[ToolConfig] = Field(
        default_factory=list,
        description="Tool configurations. Empty list exposes all registered tools.",
    )


class LLMRegistryConfig(BaseModel):
    """Registry of available LLM providers and their models."""

    model_config = ConfigDict(frozen=True)

    registry: list[LLMProviderConfig] = Field(
        min_length=1,
        description="Provider configurations. Must be non-empty.",
    )


class CompactionConfig(BaseModel):
    """Session compaction configuration."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens the LLM may generate for the compaction summary.",
        ge=1,
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Minimum log level.",
    )


class AgentConfig(BaseModel):
    """Per-agent configuration."""

    model_config = ConfigDict(frozen=True)

    llm: LLM = Field(description="Default LLM identity for this agent.")
    tools: list[Tool] = Field(
        default_factory=list,
        description="Default tools for this agent.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt overlay.",
    )
    strategy: StrategyConfig = Field(
        description="Reasoning strategy configuration.",
    )


class AgentRegistry(BaseModel):
    """Registry of agent configurations."""

    model_config = ConfigDict(frozen=True)

    agents: list[AgentConfig] = Field(
        min_length=1,
        description="Agent configurations. Must be non-empty.",
    )


class ConversationConfig(BaseModel):
    """Global configuration shared across all agents."""

    model_config = ConfigDict(frozen=True)

    llm_registry: LLMRegistryConfig = Field(
        description="Registry of available LLM providers.",
    )
    agent_registry: AgentRegistry = Field(
        description="Registry of agent configurations.",
    )
    tool_registry: ToolRegistryConfig | None = Field(
        default=None,
        description="Tool registry configuration. None means no tools are exposed.",
    )
    compaction: CompactionConfig = Field(
        default_factory=CompactionConfig,
        description="Session compaction configuration.",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration.",
    )
