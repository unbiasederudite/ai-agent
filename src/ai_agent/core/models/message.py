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


def strip_system(messages: list[Message]) -> list[Message]:
    """Remove a leading SYSTEM message if present."""
    if messages and messages[0].role == Role.SYSTEM:
        return list(messages[1:])
    return list(messages)


def prepend_system(messages: list[Message], content: str) -> list[Message]:
    """Merge content before any existing SYSTEM message, or create one.

    Used by Agent to inject identity before a compaction summary.
    """
    if messages and messages[0].role == Role.SYSTEM:
        existing = messages[0].content or ""
        merged = f"{content}\n\n{existing}" if existing else content
        return [messages[0].model_copy(update={"content": merged}), *messages[1:]]
    return [Message(role=Role.SYSTEM, content=content), *messages]


def append_system(messages: list[Message], content: str) -> list[Message]:
    """Merge content after any existing SYSTEM message, or create one.

    Used by strategies to append reasoning instructions after agent identity.
    """
    if messages and messages[0].role == Role.SYSTEM:
        existing = messages[0].content or ""
        merged = f"{existing}\n\n{content}" if existing else content
        return [messages[0].model_copy(update={"content": merged}), *messages[1:]]
    return [Message(role=Role.SYSTEM, content=content), *messages]
