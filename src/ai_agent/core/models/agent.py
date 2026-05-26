"""Agent configuration and execution state types."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_agent.core.models.llm import LLM, LLMResponse, LLMSettings
from ai_agent.core.models.message import Message
from ai_agent.core.models.strategy import StrategyConfig
from ai_agent.core.models.tool import Tool


class AgentConfig(BaseModel):
    """Configuration for a conversational agent."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Unique agent name.")
    description: str = Field(description="Human-readable agent description.")
    system_prompt: str = Field(default="", description="System prompt.")
    llm: LLM = Field(description="LLM identity.")
    settings: LLMSettings = Field(description="Sampling parameters.")
    strategy: StrategyConfig = Field(description="Reasoning strategy.")
    tools: list[Tool] = Field(
        default_factory=list,
        description="Active tools for this agent. Empty list means no tools.",
    )


class AgentStatus(StrEnum):
    """Control signal produced by a strategy at the end of each reasoning step."""

    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


class AgentState(BaseModel):
    """Immutable snapshot of the agent's runtime execution state."""

    model_config = ConfigDict(frozen=True)

    status: AgentStatus = Field(
        default=AgentStatus.RUNNING,
        description="Current control-flow status.",
    )
    turn: int = Field(
        default=0,
        ge=0,
        description="Number of strategy steps completed so far.",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="In-context conversation history.",
    )


class StepResult(BaseModel):
    """Result of one strategy step."""

    model_config = ConfigDict(frozen=True)

    state: AgentState = Field(description="New state after this step.")
    response: LLMResponse = Field(description="The LLM response that produced this step.")
