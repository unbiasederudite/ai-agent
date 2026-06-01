"""Conversational agent service."""

from __future__ import annotations

from typing import Final

from ai_agent.core.models.llm import LLMSettings
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.run import RunResult, RunSettings
from ai_agent.core.models.tool import ToolDefinition
from ai_agent.core.protocols.llm import ILLMProvider
from ai_agent.core.services.run import RunService

_BASE_PROMPT: str = """
You are a helpful AI assistant.
"""


class Agent:
    """Conversation agent."""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str | None,
        run_service: RunService,
        run_settings: RunSettings,
        base_prompt: str = _BASE_PROMPT,
    ) -> None:
        self._name: Final = name
        self._description: Final = description
        self._system_prompt: Final = "\n\n".join(p for p in (base_prompt, system_prompt) if p)
        self._run_service: Final = run_service
        self._run_settings: Final = run_settings

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def run_settings(self) -> RunSettings:
        return self._run_settings

    def run(
        self,
        messages: list[Message],
        provider: ILLMProvider,
        model: str,
        settings: LLMSettings,
        tools: list[ToolDefinition] | None,
    ) -> RunResult:
        if self._system_prompt:
            messages = [Message(role=Role.SYSTEM, content=self._system_prompt), *messages]
        return self._run_service.run(messages, provider, model, settings, tools)
