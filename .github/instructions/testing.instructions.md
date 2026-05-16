---
description: "Use when writing, reviewing, or refactoring tests. Enforces three-tier test structure, pytest conventions, naming rules, and fixture patterns for the AI agent platform."
applyTo: "**/test_*.py, **/conftest.py"
---
# Test Conventions

## Test Runner

- **pytest only.** No `unittest.TestCase`, no `nose`, no `doctest` as a test suite.

## Three-Tier Structure

Tests are split into exactly three tiers. **Confirm the correct tier before writing any test.**

```
tests/
  unit/          # Pure logic — no external deps
  mock/          # Adapter wiring — all externals mocked
  live/          # Real runtime — never in CI
```

Each tier **mirrors** the `src/` layout:

```
src/ai_agent/core/strategies/chain.py
  → tests/unit/core/strategies/test_chain.py
  → tests/mock/core/strategies/test_chain.py   (if adapter wiring tested)
  → tests/live/core/strategies/test_chain.py   (if runtime integration needed)
```

### `unit/` — Core Logic

- **No network, no filesystem, no subprocess, no external runtime of any kind.**
- No mocking of `core/` internals — if you need a mock, you are testing the wrong tier.

### `mock/` — Adapter Wiring

- **All external dependencies (LLM clients, HTTP, DB, filesystem) must be mocked.**
- Use `pytest-mock` (`mocker` fixture) or `unittest.mock.patch` as context managers — never as decorators on test functions.
- Assert on calls to mocks only at the boundary, not on internal state.

### `live/` — Real Runtime

- **Never executed in CI.** Guard every live test with:
  ```python
  pytestmark = pytest.mark.skipif(
      not os.getenv("LIVE_TESTS"), reason="opt-in: set LIVE_TESTS=1"
  )
  ```
- Every live test module **must** have a module-level docstring stating the required runtime and credentials:
  ```python
  """Live tests for OpenAI completion adapter.

  Required runtime: OpenAI API reachable.
  Required env vars: OPENAI_API_KEY.
  """
  ```
- Treat live tests as smoke tests, not exhaustive suites.

## `conftest.py` Rules

- Every folder that contains tests **must** have its own `conftest.py`, even if empty.
- Fixtures shared across a tier live in that tier's `conftest.py`.
- Fixtures shared across all tiers live in `tests/conftest.py`.
- Never import fixtures explicitly — rely on pytest's automatic discovery.

## Naming Convention

```
test_<what>_<when>_<expected>
```

Examples:
- `test_tool_selector_no_candidates_raises_selection_error`
- `test_chain_builder_max_tokens_exceeded_truncates_prompt`
- `test_config_loader_missing_key_raises_config_error`

## One Assertion Concept Per Test

- Each test verifies **one logical outcome**. Multiple `assert` statements are allowed only when they all express the same concept (e.g., checking multiple fields of a single returned model).
- Split tests that verify unrelated outcomes into separate functions.

## Test Contracts, Not Implementations

- Assert on the **public interface**: return values, raised exceptions, and observable side effects.
- Do not assert on private attributes (`_x`), internal call order, or implementation details that are not part of the contract.

## Test Data

- Use `pydantic.BaseModel` for all structured test input and expected output. **Never use raw `dict` or `tuple`.**
- Define test-specific models in the test file or in a `conftest.py` `pytest.fixture` if reused.

```python
# Correct
class ToolCallFixture(BaseModel):
    name: str
    arguments: dict[str, str]

def test_tool_call_serialises_correctly() -> None:
    fixture = ToolCallFixture(name="search", arguments={"q": "agents"})
    ...

# Wrong
def test_tool_call_serialises_correctly() -> None:
    fixture = {"name": "search", "arguments": {"q": "agents"}}
    ...
```

## Parametrize

- Use `@pytest.mark.parametrize` to cover multiple input variants of the same contract test. Each case must have an `id`.

```python
@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        pytest.param(0, True, id="zero_tokens_is_empty"),
        pytest.param(512, False, id="non_zero_tokens_not_empty"),
    ],
)
def test_budget_is_empty_returns_expected(tokens: int, expected: bool) -> None:
    ...
```

## Static Analysis

- `ruff` and `mypy --strict` must pass on test files too.
- Type-annotate all test functions with `-> None`.
- Annotate all fixture return types.
