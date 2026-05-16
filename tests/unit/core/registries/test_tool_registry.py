"""Unit tests for ToolRegistry."""

import pytest

from ai_agent.core.exceptions import ToolNotFoundError
from ai_agent.core.models.tool import Tool, ToolConfig, ToolResponse, ToolSchema
from ai_agent.core.registries import ToolRegistry


_CALC = Tool(type="test", name="calculator")
_SEARCH = Tool(type="test", name="search")

_CALC_SCHEMA = ToolSchema(
    description="Adds two numbers.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    },
)

_SEARCH_SCHEMA = ToolSchema(
    description="Searches the web.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)


class _CalcImpl:
    config: ToolConfig = ToolConfig(type="test", name="calculator")

    def execute(self, name: str, arguments: dict[str, object]) -> ToolResponse:
        a = float(arguments["a"])  # type: ignore[arg-type]
        b = float(arguments["b"])  # type: ignore[arg-type]
        return ToolResponse(content=str(a + b))


class _SearchImpl:
    config: ToolConfig = ToolConfig(type="test", name="search")

    def execute(self, name: str, arguments: dict[str, object]) -> ToolResponse:
        return ToolResponse(content="results")


class TestToolRegistryRegister:
    def test_register_valid_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CALC_SCHEMA, _CalcImpl())

    def test_register_duplicate_silently_ignored(self) -> None:
        registry = ToolRegistry()
        impl1 = _CalcImpl()
        impl2 = _CalcImpl()
        registry.register(_CALC, _CALC_SCHEMA, impl1)
        registry.register(_CALC, _CALC_SCHEMA, impl2)
        assert registry.resolve_implementation(_CALC.type) is impl1

    def test_register_multiple_tools_same_type(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CALC_SCHEMA, _CalcImpl())
        registry.register(_SEARCH, _SEARCH_SCHEMA, _SearchImpl())
        assert len(registry.tools) == 2

    def test_register_different_types_independent_implementations(self) -> None:
        other = Tool(type="rag", name="docs")
        other_schema = ToolSchema(
            description="RAG search.", parameters={"type": "object", "properties": {}}
        )

        class _RagImpl:
            config: ToolConfig = ToolConfig(type="rag", name="docs")

            def execute(self, name: str, arguments: dict[str, object]) -> ToolResponse:
                return ToolResponse(content="doc results")

        registry = ToolRegistry()
        calc_impl = _CalcImpl()
        rag_impl = _RagImpl()
        registry.register(_CALC, _CALC_SCHEMA, calc_impl)
        registry.register(other, other_schema, rag_impl)
        assert registry.resolve_implementation("test") is calc_impl
        assert registry.resolve_implementation("rag") is rag_impl


class TestToolRegistryResolveImplementation:
    def test_returns_registered_implementation(self) -> None:
        registry = ToolRegistry()
        impl = _CalcImpl()
        registry.register(_CALC, _CALC_SCHEMA, impl)
        assert registry.resolve_implementation(_CALC.type) is impl

    def test_unknown_type_raises_tool_not_found(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.resolve_implementation("nonexistent")

    def test_shared_across_same_type(self) -> None:
        registry = ToolRegistry()
        impl = _CalcImpl()
        registry.register(_CALC, _CALC_SCHEMA, impl)
        registry.register(_SEARCH, _SEARCH_SCHEMA, _SearchImpl())
        assert registry.resolve_implementation("test") is impl


class TestToolRegistryResolveSchema:
    def test_returns_registered_schema(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CALC_SCHEMA, _CalcImpl())
        assert registry.resolve_schema(_CALC) is _CALC_SCHEMA

    def test_unknown_tool_raises_tool_not_found(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.resolve_schema(_CALC)

    def test_distinct_schema_per_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CALC_SCHEMA, _CalcImpl())
        registry.register(_SEARCH, _SEARCH_SCHEMA, _SearchImpl())
        assert registry.resolve_schema(_CALC) is _CALC_SCHEMA
        assert registry.resolve_schema(_SEARCH) is _SEARCH_SCHEMA


class TestToolRegistryResolveTools:
    def test_returns_names_for_type(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CALC_SCHEMA, _CalcImpl())
        registry.register(_SEARCH, _SEARCH_SCHEMA, _SearchImpl())
        assert set(registry.resolve_tools("test")) == {"calculator", "search"}

    def test_empty_when_type_unknown(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CALC_SCHEMA, _CalcImpl())
        assert registry.resolve_tools("nonexistent") == []

    def test_excludes_other_types(self) -> None:
        other = Tool(type="rag", name="docs")
        other_schema = ToolSchema(description=".", parameters={"type": "object", "properties": {}})

        class _RagImpl:
            config: ToolConfig = ToolConfig(type="rag", name="docs")

            def execute(self, name: str, arguments: dict[str, object]) -> ToolResponse:
                return ToolResponse(content="")

        registry = ToolRegistry()
        registry.register(_CALC, _CALC_SCHEMA, _CalcImpl())
        registry.register(other, other_schema, _RagImpl())
        assert registry.resolve_tools("test") == ["calculator"]
        assert registry.resolve_tools("rag") == ["docs"]


class TestToolRegistryToolsProperty:
    def test_empty_when_no_registrations(self) -> None:
        assert ToolRegistry().tools == []

    def test_returns_all_registered_identities(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CALC_SCHEMA, _CalcImpl())
        registry.register(_SEARCH, _SEARCH_SCHEMA, _SearchImpl())
        assert set(registry.tools) == {_CALC, _SEARCH}

    def test_contains_tool_instances(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CALC_SCHEMA, _CalcImpl())
        assert all(isinstance(t, Tool) for t in registry.tools)
