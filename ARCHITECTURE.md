# Architecture

## Folder Responsibility

| Folder | Purpose |
|---|---|
| `src/ai_agent/cli/` | CLI entry points — parse argv, load config, hand off to services |
| `src/ai_agent/core/` | All agent intelligence: reasoning, decision-making, tool selection, state evolution |
| `src/ai_agent/core/exceptions/` | Full exception hierarchy rooted at `AgentError` |
| `src/ai_agent/core/factories/` | Factory functions for building core objects from configuration |
| `src/ai_agent/core/models/` | All Pydantic data models: domain models and startup config |
| `src/ai_agent/core/protocols/` | Protocol interfaces for all external providers, tools, and strategies |
| `src/ai_agent/core/registries/` | Runtime registries for agents, LLMs, and tools |
| `src/ai_agent/core/services/` | Use-case orchestration |
| `src/ai_agent/core/strategies/` | Reasoning and selection algorithm implementations |
| `src/ai_agent/core/tools/` | Concrete tool implementations |
| `src/ai_agent/adapters/` | Concrete provider adapter implementations |
| `tests/unit/` | Pure logic tests — no external deps |
| `tests/mock/` | Adapter wiring tests — all externals mocked |
| `tests/live/` | Real-runtime smoke tests — never in CI |

---

## Dependency Flow

```
cli → services → strategies → protocols → adapters
```

Cross-cutting (available to every layer): `models`, `exceptions`, `logging.Logger`

Adapters implement the protocol interfaces and therefore import from `core/protocols/`. The arrow shows the runtime call direction, not the import direction.

No circular imports. No layer calling backwards.

---

## Conversation Model

The platform supports one active conversation at a time. Session history is held in memory only — it is not persisted and is discarded when the conversation ends. A new conversation always starts from a clean state.

Within a conversation, each user request drives dynamic selection of which agent, LLM, and tools to activate. The config declares the available universe; the request determines what is used.

---

## Config System

Everything configurable is declared in a single JSON file loaded and validated at startup. Nothing can be changed after the process starts. The configurable surface is:

- **LLM registry** — available providers and their models, including sampling defaults per model
- **Agent registry** — named agents, each backed by an `AgentConfig`
- **Tool registry** — which tools are available across the system
- **Compaction** — token budget for session compaction when context grows large
- **Logging** — minimum log level

---

## Registries and Factories

Registries and factories are the binding layer between config and runtime.

- **Config declares names.** Each registry section in the config lists what is available by name.
- **Factories resolve names to instances.** At startup, `cli/` reads the config, constructs concrete adapter instances, and registers them. `core/` never constructs adapters directly.
- **Registries answer queries.** At runtime, the core requests an instance by name. If the name is not registered, a startup error is raised immediately — not at invocation time.
- **Absent names are inactive, not errors.** A name present in code but absent from config is silently unavailable. A name declared in config but not wired by a factory fails fast at startup.

---

## Exception Hierarchy

All exceptions are subclasses of `AgentError`. `core/` raises only `AgentError` subclasses. Adapters catch external exceptions and re-raise as the appropriate subclass — raw third-party exceptions must never propagate past the adapter boundary.

---

## Extension Guide

### Add a new LLM provider

`ILLMProvider` already exists — just implement it.

1. Create `XAdapter` in `adapters/` implementing `ILLMProvider`.
2. Add the provider and its models to the LLM registry section of the JSON config.
3. Wire the adapter instance in the `cli/` dependency-construction block.

No changes to any existing strategy, service, or protocol.

### Add a new provider type

Only needed when the new provider is not an LLM.

1. Define `IXProvider(Protocol)` in `core/protocols/` with the methods strategies will call.
2. Add an `XConfig` model to `core/models/config.py`.
3. Create `XAdapter` in `adapters/` implementing `IXProvider`.
4. Wire the adapter instance in the `cli/` dependency-construction block.

No changes to any existing strategy or service.

### Add a new strategy

1. Create `XStrategy` in `core/strategies/` subclassing `BaseStrategy`.
2. Subclass `StrategyConfig` with any extra fields; use `type` as the discriminator.
3. Add the class to the `implementations` dict passed to `StrategyFactory` in `cli/`.

No changes to services or adapters.

### Add a new tool

1. Create `XTool` in `core/tools/` subclassing `BaseTool`.
2. Subclass `ToolConfig` with any extra fields if the tool needs additional configuration.
3. Add the class to the `implementations` dict passed to `ToolFactory` in `cli/`.
4. Declare it in the tool registry section of the JSON config.

No changes to strategies or services.

### Add a new runtime integration

1. Create a folder `adapters/<runtime>/`.
2. Implement the runtime's entry point, translating its request/response format to `AgentState` and back.
3. Call the appropriate `XService` method — the integration owns nothing else.

No changes to `core/`.

---

## Design Decisions

**OpenAI-compatible protocol surface.**
Using the OpenAI message and tool-calling format as the internal wire format means any LLM adapter that speaks OpenAI drops in without touching `core/`. It also means the platform can act as an OpenAI-compatible endpoint itself, enabling drop-in use by existing OpenAI clients.

---

## Naming Conventions

| Pattern | Usage |
|---------|-------|
| `XStrategy` | Interchangeable algorithm implementations in `core/strategies/` |
| `XService` | Orchestration and use-case logic in `core/services/` |
| `IXProvider` | Interface / Protocol for external providers in `core/protocols/` |
| `ITool` | Interface for agent-callable tools in `core/protocols/` |
| `XAdapter` | Concrete bridge to a runtime or external system in `adapters/` |
| `XConfig` | Pydantic config model for a subsystem in `core/models/` |
| `XError` | Typed exception for a domain in `core/exceptions/` |
| `logging.getLogger(__name__)` | Standard logger (one per module) |
| Plain noun | Data-only `BaseModel` |
