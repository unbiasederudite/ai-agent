---
description: "Use when writing, reviewing, or adding documentation to Python source files or any .md file. Enforces Google docstring conventions, inline comment rules, and cross-document consistency for the AI agent platform."
applyTo: "**/*.py, **/*.md"
---
# Documentation Conventions

## Brevity Rule

Docstrings must contain only required fields — purpose, args, returns, raises. No prose explaining how the implementation works or why a design was chosen. One line per field. If a sentence can be cut without losing meaning, cut it.

## Docstrings — Required on All Public Symbols

Every public module, class, and method must have a Google-style docstring. Private members (`_name`) are exempt but encouraged for non-obvious logic.

```python
def execute(self, call: ToolCall, state: AgentState) -> ToolResult:
    """Execute a tool call against the current agent state.

    Args:
        call: Tool invocation descriptor.
        state: Full execution snapshot.

    Returns:
        ToolResult with the tool's output.

    Raises:
        ToolExecutionError: Tool raised an unhandled exception.
        ToolNotFoundError: Tool name not in the registry.
    """
```

## Inline Comments — Why, Not What

Inline comments must explain **why**, not what the code does. Delete any comment that restates the line.

```python
# Correct
# Checked here rather than in the strategy so all strategies share the same loop-detection guarantee.
if state.turn >= self._config.max_turns:
    raise LoopDetectedError(...)

# Wrong
if state.turn >= self._config.max_turns:  # check if turn limit exceeded
    raise LoopDetectedError(...)
```

## README Files — Required in Every Folder

Every `src/` folder and subfolder must contain a `README.md` that states:
1. **Purpose** — what this folder owns (one sentence).
2. **Contents** — the key modules or subfolders and what each does.

Keep it short. Update it whenever a module is added, removed, or renamed. A folder without a `README.md`, or with one that does not reflect current contents, is incomplete.

## Cross-Document Consistency

Any change to code or infrastructure must be reflected in all `.md` files that document it. A PR that changes an interface, config, or entry point without updating the relevant docs is incomplete.
