"""Conversation session owner."""

from __future__ import annotations

import logging
from typing import Any

from ai_agent.core.exceptions import ContextWindowExceededError, UserMessageTooLongError
from ai_agent.core.models.budget import ContextBudget
from ai_agent.core.models.llm import LLMUsage
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.run import RunResult, RunSettings
from ai_agent.core.models.tool import ToolDefinition
from ai_agent.core.registries.agent import AgentRegistry
from ai_agent.core.services.agent import Agent
from ai_agent.core.registries.llm import LLMRegistry
from ai_agent.core.registries.tool import ToolRegistry
from ai_agent.core.services.compaction import CompactionService

_log = logging.getLogger(__name__)


class Conversation:
    """Owns the live session state for a multi-turn conversation."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        run_settings: RunSettings,
        llm_registry: LLMRegistry,
        tool_registry: ToolRegistry,
        message_char_limit: int,
        context_budget: ContextBudget,
        compaction_service: CompactionService,
    ) -> None:
        self._agent_registry = agent_registry
        self._run_settings = run_settings
        self._llm_registry = llm_registry
        self._tool_registry = tool_registry
        self._message_char_limit = message_char_limit
        self._context_budget = context_budget
        self._compaction = compaction_service
        self._messages: list[Message] = []
        self._billed_usage = LLMUsage(input_tokens=0, output_tokens=0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def billed_usage(self) -> LLMUsage:
        return self._billed_usage

    @property
    def context_budget(self) -> ContextBudget:
        return self._context_budget

    @property
    def active_agent(self) -> Agent:
        return self._agent_registry.resolve_agent(self._run_settings.agent)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, user_message: str, settings: RunSettings | None = None) -> RunResult:
        """Send a user message and return the agent reply.

        Args:
            user_message: The user's input text.
            settings: One-shot settings override; defaults to the sticky run_settings.

        Returns:
            RunResult with the assistant reply, turn count, and token usage.

        Raises:
            UserMessageTooLongError: Message exceeds the character limit.
            ContextWindowExceededError: Context exceeded even after compaction.
        """
        if len(user_message) > self._message_char_limit:
            raise UserMessageTooLongError(
                f"User message is {len(user_message)} characters, limit is {self._message_char_limit}."
            )

        if self._context_budget.should_compact:
            self._compact()

        effective = settings if settings is not None else self._run_settings
        provider = self._llm_registry.resolve_implementation(effective.llm.provider)

        tools = self._resolve_tools(effective)

        messages = [*self._messages, Message(role=Role.USER, content=user_message)]
        try:
            result = self.active_agent.run(
                messages=messages,
                provider=provider,
                model=effective.llm.model,
                settings=effective.settings,
                tools=tools,
            )
        except ContextWindowExceededError:
            _log.warning("conversation.context_exceeded — compacting and retrying")
            self._compact()
            result = self.active_agent.run(
                messages=[*self._messages, Message(role=Role.USER, content=user_message)],
                provider=provider,
                model=effective.llm.model,
                settings=effective.settings,
                tools=tools,
            )

        self._messages.append(Message(role=Role.USER, content=user_message))
        self._messages.append(Message(role=Role.ASSISTANT, content=result.output))
        self._context_budget = self._context_budget.update(result.context_usage)
        self._billed_usage = LLMUsage(
            input_tokens=self._billed_usage.input_tokens + result.billed_usage.input_tokens,
            output_tokens=self._billed_usage.output_tokens + result.billed_usage.output_tokens,
        )

        return result

    # ------------------------------------------------------------------
    # Tool resolution
    # ------------------------------------------------------------------

    def _resolve_tools(self, settings: RunSettings) -> list[ToolDefinition] | None:
        if settings.tools is None:
            return [self._tool_registry.resolve_definition(t) for t in self._tool_registry.tools]
        if len(settings.tools) == 0:
            return None
        return [self._tool_registry.resolve_definition(t) for t in settings.tools]

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def _compact(self) -> None:
        result = self._compaction.compact(self._messages)
        self._messages = list(result.messages)
        self._context_budget = self._context_budget.reset_usage()
        if result.usage:
            self._billed_usage = LLMUsage(
                input_tokens=self._billed_usage.input_tokens + result.usage.input_tokens,
                output_tokens=self._billed_usage.output_tokens + result.usage.output_tokens,
            )

    # ------------------------------------------------------------------
    # Settings mutation
    # ------------------------------------------------------------------

    def sticky(self, field: str, value: Any) -> None:
        """Permanently update a RunSettings field for all future runs.

        Args:
            field: RunSettings field name to update.
            value: New value for the field.
        """
        if field == "agent":
            self._run_settings = self._agent_registry.resolve_agent(value).run_settings
            provider = self._llm_registry.resolve_implementation(self._run_settings.llm.provider)
            self._context_budget = self._context_budget.recalibrate(
                context_window=provider.context_window(self._run_settings.llm.model)
            )
        elif field == "llm":
            new_settings = self._llm_registry.resolve_settings(value)
            self._run_settings = self._run_settings.model_copy(
                update={"llm": value, "settings": new_settings}
            )
            provider = self._llm_registry.resolve_implementation(value.provider)
            self._context_budget = self._context_budget.recalibrate(
                context_window=provider.context_window(value.model)
            )
        else:
            self._run_settings = self._run_settings.model_copy(update={field: value})

    def loose(self, field: str, value: Any) -> RunSettings:
        """Return a one-shot RunSettings copy with one field changed.

        Args:
            field: RunSettings field name to update.
            value: New value for the field.

        Returns:
            A new RunSettings with the specified field overridden.
        """
        return self._run_settings.model_copy(update={field: value})

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear message history and accumulated usage."""
        self._messages = []
        self._billed_usage = LLMUsage(input_tokens=0, output_tokens=0)
        self._context_budget = self._context_budget.reset_usage()
