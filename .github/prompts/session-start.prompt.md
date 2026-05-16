---
description: "Run once at the start of every new chat session. Orients the agent, confirms the project state, and declares the active task."
name: "Session Start"
agent: "agent"
tools: [read, search, execute]
argument-hint: "Describe what you want to work on in this session"
---
You are starting a new session on the AI agent platform. Work through the following steps before responding with anything else.

## 1. Load Context Map

Read: [ARCHITECTURE.md](../../ARCHITECTURE.md)

## 2. Confirm Baseline Health

Run all checks from the **Code Checks** section of [`README.md`](../../README.md) (`--tb=no -q` for pytest) and report results. If any command fails, identify the issue and inform the user. Do not proceed past this step until all commands pass — do not continue with broken code.

## 3. Clarify the Session Goal

If the user provided an argument (the task for this session), restate it in one sentence and confirm the intended agent or workflow:
- New feature → use the **Feature** agent
- Code review → use the **Review** agent
- Context loading only → you are done

If no argument was provided, ask: *"What would you like to work on in this session?"*

Do not write any code until the session goal is confirmed.
