"""IAgent Protocol."""

from typing import Protocol, runtime_checkable

from ai_agent.core.models.llm import LLMSettings
from ai_agent.core.models.message import Message
from ai_agent.core.models.run import RunResult, RunSettings
from ai_agent.core.models.tool import ToolDefinition
from ai_agent.core.protocols.llm import ILLMProvider


@runtime_checkable
class IAgent(Protocol):
    """Interface that every agent implementation must satisfy."""

    @property
    def run_settings(self) -> RunSettings:
        """Default run settings for this agent."""
        ...

    def run(
        self,
        messages: list[Message],
        provider: ILLMProvider,
        model: str,
        settings: LLMSettings,
        tools: list[ToolDefinition] | None,
    ) -> RunResult:
        """Execute a full reasoning run.

        Args:
            messages: Message history for this run.
            provider: LLM provider.
            model: Model identifier.
            settings: Sampling parameters.
            tools: Tool definitions, or None.

        Returns:
            Run result.
        """
        ...
