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

    def context_window(self, model: str) -> int:
        """Return the maximum number of input tokens the given model accepts.

        Args:
            model: Model identifier string.

        Returns:
            A positive integer representing the model's context window in tokens.
        """
        ...
