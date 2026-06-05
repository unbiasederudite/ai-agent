# core/strategies

Concrete reasoning strategy implementations.

The `IReasoningStrategy` Protocol lives in `core/protocols/strategy.py`.

## Files

- `__init__.py` — re-exports `BaseStrategy` and `ReActStrategy`.
- `base.py` — `BaseStrategy`: shared base class for all strategy implementations.
- `react.py` — `ReActStrategy`: ReAct (Reason + Act) strategy. One LLM call per step; dispatches tool calls when present, otherwise completes.
