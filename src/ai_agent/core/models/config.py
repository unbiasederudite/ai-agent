"""Configuration models for the AI agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_agent.core.models.agent import Agent, AgentConfig
from ai_agent.core.models.llm import LLM, LLMSettings
from ai_agent.core.models.tool import ToolConfig


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

    llm: LLM = Field(description="LLM used to generate compaction summaries.")
    temperature: float = Field(
        description="Sampling temperature for the compaction LLM.", ge=0.0, le=2.0
    )
    threshold: float = Field(
        description="Context fill fraction that triggers compaction.", ge=0.0, le=1.0
    )
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


class AgentRegistryConfig(BaseModel):
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
    agent_registry: AgentRegistryConfig = Field(
        description="Registry of agent configurations.",
    )
    default_agent: Agent = Field(
        description="Active agent at conversation start. Must be present in agent_registry.",
    )
    tool_registry: ToolRegistryConfig | None = Field(
        default=None,
        description="Tool registry configuration. None means no tools are exposed.",
    )
    compaction: CompactionConfig = Field(
        description="Session compaction configuration.",
    )
    message_char_limit: int = Field(
        default=10_000,
        ge=1,
        description="Maximum character length of a user message.",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration.",
    )

    @model_validator(mode="after")
    def _validate_references(self) -> ConversationConfig:
        agent_names = {a.name for a in self.agent_registry.agents}
        if self.default_agent.name not in agent_names:
            raise ValueError(f"default_agent {self.default_agent.name!r} is not in agent_registry.")
        registered_llms = {
            (p.provider, m.model) for p in self.llm_registry.registry for m in p.models
        }
        if (self.compaction.llm.provider, self.compaction.llm.model) not in registered_llms:
            raise ValueError(f"compaction.llm {self.compaction.llm!r} is not in llm_registry.")
        return self
