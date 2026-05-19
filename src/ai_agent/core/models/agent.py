"""Agent identity and execution state types."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_agent.core.models.llm import LLMResponse
from ai_agent.core.models.message import Message


class Agent(BaseModel):
    """Agent identity."""

    model_config = ConfigDict(frozen=True)

    type: str = Field(description="Agent type identifier.")
    name: str = Field(description="Unique agent name.")


class AgentConfig(Agent):
    """Base configuration for all agent types. Subclass with concrete fields per implementation."""


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
