"""Tool identity, schema, invocation, and result models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ToolSchema(BaseModel):
    """Description of a tool exposed to the LLM."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(description="Human-readable description of what the tool does.")
    parameters: dict[str, object] = Field(description="JSON Schema for input parameters.")


class Tool(BaseModel):
    """Tool identity."""

    model_config = ConfigDict(frozen=True)

    type: str = Field(description="Tool type.")
    name: str = Field(description="Tool name.")


class ToolConfig(Tool):
    """Base settings for all tool types.

    Concrete settings subclass this with a Literal 'type' field and add their own fields.
    When multiple types exist, replace this class with a discriminated union:

        ToolConfig = Annotated[Union[FooSettings, BarSettings], Field(discriminator="type")]
    """


class ToolDefinition(BaseModel):
    """Named tool definition sent to the LLM."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Tool name.")
    tool_schema: ToolSchema = Field(description="Tool schema.")


class ToolContext(BaseModel):
    """Shared correlation fields for tool interactions."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique identifier for this tool interaction.")
    name: str = Field(description="Name of the tool.")


class ToolCall(ToolContext):
    """A single tool invocation issued by the agent."""

    arguments: dict[str, object] = Field(
        default_factory=dict,
        description="Keyword arguments for the tool.",
    )


class ToolResponse(BaseModel):
    """Raw output from a tool execution."""

    model_config = ConfigDict(frozen=True)

    content: str | dict[str, object] = Field(
        description="Output returned by the tool.",
    )
    is_error: bool = Field(
        default=False,
        description="True when the tool reported a failure.",
    )


class ToolResult(ToolContext, ToolResponse):
    """Tool response with correlation fields attached by the service layer."""
