---
description: "Use when writing, reviewing, or refactoring Python source files. Enforces type safety, naming conventions, logging, exception handling, and architecture rules for the AI agent platform."
applyTo: "**/*.py"
---
# Python Code Conventions

## Type Annotations

All parameters, return types, and attributes must be fully type-annotated. No implicit `Any`.

## Exception Handling

- **No bare `except:`** and no `except Exception:` without re-raising or logging at ERROR level.
- Define typed exception classes (subclass `AgentError` or a domain-specific base) for every error boundary.

## Logging

- **No `print()`** anywhere in the codebase.
- Use `logging.getLogger(__name__)` at module level. No `print()` anywhere.

## OOP and Module Structure

- **All logic lives in classes.** Free-standing functions are not permitted in any layer. The only module-level callables allowed are CLI entry-point decorators and test functions.
- Module-level code is limited to: imports, `__all__`, type aliases, and constants.
- Every public package (`__init__.py`) must declare `__all__`. No `import *` anywhere.

```python
# Correct
class PromptBuilder:
    def build(self, messages: list[Message], config: ReasoningConfig) -> str:
        return self._truncate(self._format(messages), config.max_tokens)

    def _format(self, messages: list[Message]) -> str: ...
    def _truncate(self, text: str, max_tokens: int) -> str: ...

# Wrong — free-standing function
def truncate_prompt(text: str, max_tokens: int) -> str: ...
```

## Dependency Injection

- **Constructor injection only.** Every dependency a class needs must be passed as a constructor parameter and stored as a private annotated attribute. No globals, no `importlib` lookups, no service-locator pattern.
- Use `Final` for injected dependencies that must not be reassigned after construction.

## Data Models

All cross-boundary data structures use Pydantic `BaseModel`. No dataclasses, no `TypedDict`, no plain `dict` at any boundary.

## Async Conventions

- **Sync-first.** Every public method starts as synchronous. Add `async def` variants only when a caller explicitly requires async execution.
- **No `asyncio.run()` in library code.** Event loop ownership belongs exclusively to the CLI entry point or the framework adapter. `core/` and `adapters/` must never call `asyncio.run()`, `asyncio.get_event_loop()`, or `loop.run_until_complete()`.
