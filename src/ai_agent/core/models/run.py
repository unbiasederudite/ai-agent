"""Run-level models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ai_agent.core.models.llm import LLM, LLMSettings, LLMUsage
from ai_agent.core.models.message import Message
from ai_agent.core.models.tool import Tool


class RunSettings(BaseModel):
    """Per-conversation LLM settings."""

    model_config = ConfigDict(frozen=True)

    agent: str = Field(description="Active agent name.")
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
    billed_usage: LLMUsage = Field(description="Accumulated token counts across all turns.")
    context_usage: LLMUsage = Field(description="Token counts for the final turn.")
    messages: list[Message] = Field(
        description="Full message trace including all tool calls and results."
    )
