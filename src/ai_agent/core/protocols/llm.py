"""ILLMProvider Protocol."""

from typing import Protocol, runtime_checkable

from ai_agent.core.models.llm import LLMRequest, LLMResponse


@runtime_checkable
class ILLMProvider(Protocol):
    """Interface that every LLM provider adapter must satisfy."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to the LLM.

        Args:
            request: The request including model, messages, and optional tools.

        Returns:
            A validated LLMResponse.
        """
        ...
