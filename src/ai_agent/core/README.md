# core

All agent intelligence and business logic. Contains no adapter or CLI imports.

## Sub-packages

| Package | Description |
|---------|-------------|
| `context/` | Token budgeting, session history, and compaction |
| `exceptions/` | Platform exception hierarchy |
| `models/` | Shared Pydantic data models and configuration models |
| `protocols/` | All Protocol interfaces |
| `registries/` | Runtime registries for agents, LLMs, and tools |
| `services/` | Use-case orchestration services |
| `strategies/` | Concrete reasoning strategy implementations |

## Files

- `__init__.py` — package marker, no exports.
- `agent.py` — `AgentNode`, the runtime binding of a system prompt, `RunSettings`, and `RunService`. Prepends the system prompt on each `run()` call.
