"""Run-level models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ai_agent.core.models.llm import LLM, LLMSettings, LLMUsage
from ai_agent.core.models.tool import Tool


class RunSettings(BaseModel):
    """Per-conversation LLM settings."""

    model_config = ConfigDict(frozen=True)

    llm: LLM = Field(description="LLM identity.")
    settings: LLMSettings = Field(description="Sampling parameters.")
    tools: list[Tool] | None = Field(
        default=None,
        description="Active tools. None = all registered; [] = disabled.",
    )


class RunResult(BaseModel):
    """Result of a single agent run."""

    model_config = ConfigDict(frozen=True)

    output: str = Field(description="Text content of the last assistant message.")
    turns: int = Field(description="Number of reasoning turns executed.", ge=0)
    usage: LLMUsage = Field(description="Token counts for this run.")
