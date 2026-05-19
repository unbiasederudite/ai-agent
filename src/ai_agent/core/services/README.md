# core/services

Use-case orchestration services.

## Files

- `__init__.py` — exports `CompactionService`, `RunService`, and `ToolService`.
- `compaction.py` — `CompactionService`, calls an LLM to summarise session history into a single `SYSTEM`-role `CompactionResult`. Used by strategies that need to compact the context window before calling the LLM.
- `run.py` — `RunService`, a pure reasoning loop runner. Takes a single `IReasoningStrategy` at construction; `run()` calls `strategy.step()` each turn and exits only on `COMPLETE` or `ERROR`. Returns `billed_usage` (accumulated across all turns) and `context_usage` (final turn only). Has no tool knowledge — all state transitions beyond those two are the strategy's concern.
- `tool.py` — `ToolService`, executes a list of `ToolCall` objects against a `ToolRegistry` and returns one `ToolResult` per call. Unknown tools and execution failures both produce an error result rather than raising. Lives at the conversation layer and is injected into strategies that need it.
