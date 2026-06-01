# core/strategies

Concrete reasoning strategy implementations.

The `IReasoningStrategy` Protocol lives in `core/protocols/strategy.py`.

## Files

- `__init__.py` — re-exports `BaseStrategy` and `IReasoningStrategy`.
- `base.py` — `BaseStrategy`: shared base class for all strategy implementations.

Concrete strategy implementations are added here as individual modules, subclassing `BaseStrategy`.
