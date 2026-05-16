"""Unit tests for Tool, ToolSchema, and ToolDefinition models."""

import pytest

from ai_agent.core.models.tool import Tool, ToolDefinition, ToolSchema


class TestTool:
    def test_tool_constructs(self) -> None:
        tool = Tool(type="mcp", name="calculator")
        assert tool.type == "mcp"
        assert tool.name == "calculator"

    def test_tool_is_frozen(self) -> None:
        tool = Tool(type="mcp", name="calculator")
        with pytest.raises(Exception):
            tool.name = "other"  # type: ignore[misc]

    def test_tool_requires_type(self) -> None:
        with pytest.raises(Exception):
            Tool(name="calculator")  # type: ignore[call-arg]

    def test_tool_requires_name(self) -> None:
        with pytest.raises(Exception):
            Tool(type="mcp")  # type: ignore[call-arg]


class TestToolSchema:
    def test_tool_schema_constructs(self) -> None:
        schema = ToolSchema(
            description="Performs arithmetic calculations.",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        )
        assert schema.description == "Performs arithmetic calculations."

    def test_tool_schema_is_frozen(self) -> None:
        schema = ToolSchema(
            description="No-op tool.",
            parameters={"type": "object", "properties": {}},
        )
        with pytest.raises(Exception):
            schema.description = "changed"  # type: ignore[misc]

    def test_tool_schema_parameters_is_json_schema_dict(self) -> None:
        schema = ToolSchema(
            description="Searches the web.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        assert schema.parameters["type"] == "object"

    def test_tool_schema_has_no_name_field(self) -> None:
        assert "name" not in ToolSchema.model_fields

    def test_tool_schema_has_no_strict_field(self) -> None:
        assert "strict" not in ToolSchema.model_fields


class TestToolDefinition:
    def _make_schema(self) -> ToolSchema:
        return ToolSchema(
            description="Performs arithmetic.",
            parameters={"type": "object", "properties": {}},
        )

    def test_tool_definition_constructs(self) -> None:
        defn = ToolDefinition(name="calculator", tool_schema=self._make_schema())
        assert defn.name == "calculator"
        assert isinstance(defn.tool_schema, ToolSchema)

    def test_tool_definition_is_frozen(self) -> None:
        defn = ToolDefinition(name="calculator", tool_schema=self._make_schema())
        with pytest.raises(Exception):
            defn.name = "other"  # type: ignore[misc]

    def test_tool_definition_requires_name(self) -> None:
        with pytest.raises(Exception):
            ToolDefinition(tool_schema=self._make_schema())  # type: ignore[call-arg]

    def test_tool_definition_requires_schema(self) -> None:
        with pytest.raises(Exception):
            ToolDefinition(name="calculator")  # type: ignore[call-arg]
