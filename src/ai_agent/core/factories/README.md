# core/factories

Factory functions for building core objects from configuration.

## Files

- `__init__.py` — re-exports all factories: `AgentFactory`, `ConversationFactory`, `LLMFactory`, `StrategyFactory`, `ToolFactory`.
- `agent.py` — `AgentFactory`: builds `Agent` instances from `AgentConfig`.
- `conversation.py` — `ConversationFactory`: builds a fully wired `Conversation` from `ConversationConfig`.
- `llm.py` — `LLMFactory`: resolves provider names to pre-built `ILLMProvider` instances.
- `strategy.py` — `StrategyFactory`: instantiates a concrete `IReasoningStrategy` from a `BaseStrategyConfig`.
- `tool.py` — `ToolFactory`: instantiates a concrete `ITool` from a `BaseToolConfig`.
