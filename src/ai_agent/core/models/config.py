"""Configuration models for the AI agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_agent.core.models.agent import AgentConfig
from ai_agent.core.models.llm import LLM, LLMSettings
from ai_agent.core.models.tool import BaseToolConfig


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


class CompactionConfig(BaseModel):
    """Session compaction configuration."""

    model_config = ConfigDict(frozen=True)

    llm: LLM = Field(description="LLM used to generate compaction summaries.")
    settings: LLMSettings = Field(description="Sampling parameters for the compaction LLM.")
    threshold: float = Field(
        description="Context fill fraction that triggers compaction.", ge=0.0, le=1.0
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Minimum log level.",
    )


class ConversationConfig(BaseModel):
    """Global configuration shared across all agents."""

    model_config = ConfigDict(frozen=True)

    llm_registry: list[LLMProviderConfig] = Field(
        min_length=1,
        description="Available LLM providers and their models.",
    )
    agent_registry: list[AgentConfig] = Field(
        min_length=1,
        description="Agent configurations.",
    )
    default_agent: str = Field(
        description="Name of the active agent at conversation start. Must be present in agent_registry.",
    )
    tool_registry: list[BaseToolConfig] = Field(
        default_factory=list,
        description="Tool configurations exposed to the LLM. Empty list means no tools.",
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
        agent_names = {a.name for a in self.agent_registry}
        if self.default_agent not in agent_names:
            raise ValueError(f"default_agent {self.default_agent!r} not in agent_registry.")

        registered_llms = {(p.provider, m.model) for p in self.llm_registry for m in p.models}

        if (self.compaction.llm.provider, self.compaction.llm.model) not in registered_llms:
            raise ValueError(f"compaction.llm {self.compaction.llm!r} not in llm_registry.")

        registered_tools = {t.name for t in self.tool_registry}

        for agent in self.agent_registry:
            if (agent.llm.provider, agent.llm.model) not in registered_llms:
                raise ValueError(f"Agent {agent.name!r} llm {agent.llm!r} not in llm_registry.")
            if agent.tools:
                for tool in agent.tools:
                    if tool.name not in registered_tools:
                        raise ValueError(
                            f"Agent {agent.name!r} references unknown tool {tool.name!r}."
                        )

        return self
