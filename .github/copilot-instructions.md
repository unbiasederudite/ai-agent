# AI Agent Platform — Copilot Guidelines

## Project Context

A modular, configurable AI agent platform built as a pure Python monolith core. The core owns all agent intelligence; frameworks are plug-in adapters that handle execution mechanics only. Everything is configurable at startup via JSON. The protocol surface is OpenAI-compatible.

**Core stack**: Python 3.13 · uv · Pydantic v2 · ruff · mypy strict

## Context Map — Read Before Touching Code

On every new task or context refresh, read the files listed in the **AI Agent Reading Sequence** section of [`README.md`](../README.md), in order, before writing or suggesting any code.

## Agent Behavior Rules

### Ask Clarifying Questions
- If implementation details are missing or ambiguous, ask targeted questions about those specifics before proceeding.

### Ask Before Implementing New Features
- Present the optimal solution(s) to the problem and wait for explicit approval before writing code.
- If uncertain between two valid approaches, present both with tradeoffs — do not pick silently.

### Test-Driven Development (Preferred)
- For non-trivial logic: propose the test contract first, then implement.
- Trivial helpers (pure transforms, single-expression utilities) may skip the TDD ceremony.
- Never leave a non-trivial implementation without a corresponding `unit/` test.

### Scope Discipline
- One feature at a time. Flag scope creep immediately if a request implies changes beyond the stated goal.
- Prefer the smallest change that satisfies the requirement — do not refactor unrelated code.
- Prefer editing existing files over creating new ones.

### Self-Review
After every new feature implementation, check:
- [ ] `ruff` and `mypy --strict` would pass
- [ ] A test suite for the feature exists in the tests folder
- [ ] All `.md` files that document the affected feature or component are up to date

### Flag Assumptions
State any assumption made during generation explicitly. If an assumption is load-bearing (affects the design), ask for confirmation before proceeding.
