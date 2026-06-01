"""Unit tests for ToolRegistry."""

import pytest

from ai_agent.core.exceptions import ToolNotFoundError
from ai_agent.core.models.tool import Tool, ToolResponse, ToolSchema
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
    @property
    def schema(self) -> ToolSchema:
        return _CALC_SCHEMA

    def execute(self, arguments: dict[str, object]) -> ToolResponse:
        a = float(arguments["a"])  # type: ignore[arg-type]
        b = float(arguments["b"])  # type: ignore[arg-type]
        return ToolResponse(content=str(a + b))


class _SearchImpl:
    @property
    def schema(self) -> ToolSchema:
        return _SEARCH_SCHEMA

    def execute(self, arguments: dict[str, object]) -> ToolResponse:
        return ToolResponse(content="results")


class TestToolRegistryRegister:
    def test_register_valid_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CalcImpl())

    def test_register_duplicate_silently_ignored(self) -> None:
        registry = ToolRegistry()
        impl1 = _CalcImpl()
        impl2 = _CalcImpl()
        registry.register(_CALC, impl1)
        registry.register(_CALC, impl2)
        assert registry.resolve_implementation(_CALC) is impl1

    def test_register_multiple_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CalcImpl())
        registry.register(_SEARCH, _SearchImpl())
        assert len(registry.tools) == 2

    def test_different_tools_have_independent_implementations(self) -> None:
        other = Tool(type="rag", name="docs")

        class _RagImpl:
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    description="RAG.", parameters={"type": "object", "properties": {}}
                )

            def execute(self, arguments: dict[str, object]) -> ToolResponse:
                return ToolResponse(content="doc results")

        registry = ToolRegistry()
        calc_impl = _CalcImpl()
        rag_impl = _RagImpl()
        registry.register(_CALC, calc_impl)
        registry.register(other, rag_impl)
        assert registry.resolve_implementation(_CALC) is calc_impl
        assert registry.resolve_implementation(other) is rag_impl


class TestToolRegistryResolveImplementation:
    def test_returns_registered_implementation(self) -> None:
        registry = ToolRegistry()
        impl = _CalcImpl()
        registry.register(_CALC, impl)
        assert registry.resolve_implementation(_CALC) is impl

    def test_unknown_tool_raises_tool_not_found(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.resolve_implementation(Tool(type="nonexistent", name="x"))

    def test_each_tool_resolves_to_its_own_implementation(self) -> None:
        registry = ToolRegistry()
        calc_impl = _CalcImpl()
        search_impl = _SearchImpl()
        registry.register(_CALC, calc_impl)
        registry.register(_SEARCH, search_impl)
        assert registry.resolve_implementation(_CALC) is calc_impl
        assert registry.resolve_implementation(_SEARCH) is search_impl


class TestToolRegistryResolveDefinition:
    def test_returns_definition_with_correct_name(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CalcImpl())
        assert registry.resolve_definition(_CALC).name == "calculator"

    def test_returns_definition_with_schema_from_implementation(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CalcImpl())
        assert registry.resolve_definition(_CALC).tool_schema is _CALC_SCHEMA

    def test_unknown_tool_raises_tool_not_found(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.resolve_definition(_CALC)

    def test_distinct_definition_per_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CalcImpl())
        registry.register(_SEARCH, _SearchImpl())
        assert registry.resolve_definition(_CALC).tool_schema is _CALC_SCHEMA
        assert registry.resolve_definition(_SEARCH).tool_schema is _SEARCH_SCHEMA


class TestToolRegistryResolveTools:
    def test_returns_names_for_type(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CalcImpl())
        registry.register(_SEARCH, _SearchImpl())
        assert set(registry.resolve_tools("test")) == {"calculator", "search"}

    def test_empty_when_type_unknown(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CalcImpl())
        assert registry.resolve_tools("nonexistent") == []

    def test_excludes_other_types(self) -> None:
        other = Tool(type="rag", name="docs")

        class _RagImpl:
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(description=".", parameters={"type": "object", "properties": {}})

            def execute(self, arguments: dict[str, object]) -> ToolResponse:
                return ToolResponse(content="")

        registry = ToolRegistry()
        registry.register(_CALC, _CalcImpl())
        registry.register(other, _RagImpl())
        assert registry.resolve_tools("test") == ["calculator"]
        assert registry.resolve_tools("rag") == ["docs"]


class TestToolRegistryToolsProperty:
    def test_empty_when_no_registrations(self) -> None:
        assert ToolRegistry().tools == []

    def test_returns_all_registered_identities(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CalcImpl())
        registry.register(_SEARCH, _SearchImpl())
        assert set(registry.tools) == {_CALC, _SEARCH}

    def test_contains_tool_instances(self) -> None:
        registry = ToolRegistry()
        registry.register(_CALC, _CalcImpl())
        assert all(isinstance(t, Tool) for t in registry.tools)
