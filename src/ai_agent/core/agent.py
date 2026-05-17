"""AgentNode — runtime binding of a system prompt, RunSettings, and RunService."""

from __future__ import annotations

from ai_agent.core.models.llm import LLMSettings
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.run import RunResult, RunSettings
from ai_agent.core.models.tool import ToolDefinition
from ai_agent.core.protocols.llm import ILLMProvider
from ai_agent.core.services.run import RunService


class AgentNode:
    """Pairs a system prompt and RunSettings with a RunService.

    AgentConfig is only needed at build time; AgentNode holds the minimal
    runtime state — system prompt, sampling defaults, and the reasoning loop.
    """

    def __init__(self, system_prompt: str, service: RunService, run_settings: RunSettings) -> None:
        self._system_prompt = system_prompt
        self._service = service
        self._run_settings = run_settings

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
        """Prepend the system prompt and delegate to the RunService."""
        full_messages = [Message(role=Role.SYSTEM, content=self._system_prompt), *messages]
        return self._service.run(full_messages, provider, model, settings, tools)
