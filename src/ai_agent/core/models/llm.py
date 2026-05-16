"""LLM request and response models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.tool import ToolDefinition


class FinishReason(StrEnum):
    """Reason the LLM stopped generating."""

    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"


class LLM(BaseModel):
    """Identifies an LLM by provider and model name."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(description="Provider name.")
    model: str = Field(description="Model identifier.")


class LLMUsage(BaseModel):
    """Token counts for one LLM completion."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(description="Input token count.", ge=0)
    output_tokens: int = Field(description="Output token count.", ge=0)


class LLMSettings(BaseModel):
    """LLM sampling parameters."""

    model_config = ConfigDict(frozen=True)

    temperature: float = Field(description="Sampling temperature.", ge=0.0, le=2.0)
    max_tokens: int = Field(description="Maximum tokens to generate.", ge=1)


class LLMRequest(BaseModel):
    """Request sent to an LLM provider."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(description="Model identifier.")
    settings: LLMSettings = Field(description="Sampling parameters.")
    messages: list[Message] = Field(description="Conversation history.", min_length=1)
    tools: list[ToolDefinition] | None = Field(
        default=None,
        description="Tool definitions. None disables tool calling.",
    )


class LLMResponse(BaseModel):
    """Response received from an LLM provider."""

    model_config = ConfigDict(frozen=True)

    message: Message = Field(description="The assistant message.")
    finish_reason: FinishReason = Field(description="Why the model stopped generating.")
    usage: LLMUsage = Field(description="Token counts for this completion.")

    @model_validator(mode="after")
    def message_must_be_assistant(self) -> LLMResponse:
        """Ensure the response message always has role ASSISTANT."""
        if self.message.role != Role.ASSISTANT:
            raise ValueError(
                f"LLMResponse.message must have role 'assistant', got '{self.message.role}'."
            )
        return self
