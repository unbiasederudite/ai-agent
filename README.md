# AI Agent Platform

A modular, configurable AI agent platform built as a pure Python monolith core. The core owns all agent intelligence — reasoning strategies, decision-making, tool selection, and state evolution. Frameworks are plug-in adapters that handle execution mechanics only. Everything is configurable at startup via a JSON config file. The protocol surface is OpenAI-compatible.

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

## Install

```bash
uv sync
uv run pre-commit install    # once after clone — wires ruff + mypy as git hooks
```

## Run

```bash
uv run ai-agent --config path/to/config.json
```

## Code Checks

All checks must pass before merging. Run them in this order:

```bash
uv run ruff check .                          # lint
uv run ruff format --check .                 # format (check only, no write)
uv run mypy --strict src/                    # type-check
uv run pytest tests/unit/ tests/mock/ -q     # CI-eligible tests
LIVE_TESTS=1 uv run pytest tests/live/ -q    # real-runtime tests (opt-in, never in CI)
```

## AI Agent Reading Sequence

Read these files in order before writing or reviewing any code:

1. [`ARCHITECTURE.md`](ARCHITECTURE.md) — component map, dependency flow, naming conventions, design decisions
2. [`.github/instructions/python.instructions.md`](.github/instructions/python.instructions.md) — type hints, logging, exceptions, OOP, DI, async
3. [`.github/instructions/docs.instructions.md`](.github/instructions/docs.instructions.md) — docstring conventions, cross-doc consistency
4. [`.github/instructions/testing.instructions.md`](.github/instructions/testing.instructions.md) — three-tier test structure, pytest rules
5. [`.github/instructions/dependencies.instructions.md`](.github/instructions/dependencies.instructions.md) — uv only, runtime vs dev, version pinning
6. [`pyproject.toml`](pyproject.toml) — declared dependencies and entry points

## Project Structure

```
src/ai_agent/
  cli/            # Entry points — parse argv, load config, delegate to services
  core/
    exceptions/   # AgentError hierarchy
    models/       # All Pydantic data models
    protocols/    # All Protocol interfaces
    registries/   # Runtime registries
    services/     # Use-case orchestration
    strategies/   # Reasoning and selection algorithm implementations
    context/      # Token budgeting, LLM-based compaction
  adapters/       # Concrete provider adapter implementations

tests/
  unit/           # Pure logic — no external deps, mirrors src/ layout
  mock/           # Adapter wiring — all externals mocked, mirrors src/ layout
  live/           # Real-runtime smoke tests — never in CI, opt-in via LIVE_TESTS=1
```
