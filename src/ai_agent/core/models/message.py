"""Conversation message model."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_agent.core.models.tool import ToolCall


class Role(StrEnum):
    """Valid message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """An immutable chat message."""

    model_config = ConfigDict(frozen=True)

    role: Role = Field(description="The speaker role.")
    content: str | None = Field(
        default=None,
        description="Text content. May be None for assistant messages that only contain tool calls.",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="ID of the ToolCall this message responds to.",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="Tool invocations issued by the assistant.",
    )
