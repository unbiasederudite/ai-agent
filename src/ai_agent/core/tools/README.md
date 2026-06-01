# core/tools

Concrete tool implementations.

The `ITool` Protocol lives in `core/protocols/tool.py`.

## Files

- `__init__.py` — re-exports `BaseTool` and `ITool`.
- `base.py` — `BaseTool`: shared base class for all tool implementations.

Concrete tool implementations are added here as individual modules, subclassing `BaseTool`.
