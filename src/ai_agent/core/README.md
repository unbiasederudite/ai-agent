# core

All agent intelligence and business logic. Contains no adapter or CLI imports.

## Sub-packages

| Package | Description |
|---------|-------------|
| `context/` | Token budgeting, session history, and compaction |
| `exceptions/` | Platform exception hierarchy |
| `models/` | Shared Pydantic data models and configuration models |
| `protocols/` | All Protocol interfaces |
| `registries/` | Runtime registries for LLMs and tools |
| `services/` | Agent execution service |
| `strategies/` | Concrete reasoning strategy implementations |

## Files

- `__init__.py` — package marker, no exports.
