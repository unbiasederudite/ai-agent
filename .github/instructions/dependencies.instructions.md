---
description: "Use when adding, removing, or updating dependencies in pyproject.toml, or when deciding whether a new library is needed. Enforces dependency hygiene for the AI agent platform."
applyTo: "**/pyproject.toml, **/*.py"
---
# Dependency Conventions

## Package Manager

- **`uv` only.** Never use `pip`, `pip-tools`, `poetry`, or `conda` to install, remove, or lock dependencies.
- All dependency operations go through `uv`:
  - Add runtime dep: `uv add <package>`
  - Add dev dep: `uv add --dev <package>`
  - Remove dep: `uv remove <package>`
  - Sync environment: `uv sync`
- `pyproject.toml` is the single source of truth. Never manually edit `uv.lock`.

## Runtime vs Dev Dependencies

| Belongs in `dependencies` | Belongs in `dev-dependencies` |
|---|---|
| Libraries imported in `src/` at runtime | Testing libraries (`pytest`, `pytest-mock`, …) |
| Pydantic, Typer, and other core stack libs | Linters and formatters (`ruff`) |
| Adapter libraries used at runtime | Type checkers (`mypy`) |
| | `pre-commit` |

A library needed only in `tests/` or `.github/` is always a dev dependency.

## Version Pinning Policy

- Use a **minimum version lower-bound** for runtime dependencies: `>=X.Y`.
- Use a **compatible-release** constraint when the library uses semantic versioning strictly: `~=X.Y`.
- Use an **exact pin** only for packages with a known breaking history or for security-sensitive libraries.
- Never use an unbounded `*` version specifier for a runtime dependency.
