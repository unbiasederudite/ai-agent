"""IReasoningStrategy Protocol."""

from typing import Protocol, runtime_checkable

from ai_agent.core.models.llm import LLMRequest
from ai_agent.core.models.agent import AgentState, StepResult
from ai_agent.core.models.strategy import StrategyConfig
from ai_agent.core.protocols.llm import ILLMProvider


@runtime_checkable
class IReasoningStrategy(Protocol):
    """Interface that every reasoning strategy must satisfy."""

    config: StrategyConfig

    def step(
        self,
        state: AgentState,
        request: LLMRequest,
        provider: ILLMProvider,
    ) -> StepResult:
        """Execute one reasoning step.

        Args:
            state: Current execution snapshot.
            request: LLM request for this step.
            provider: LLM provider to call.

        Returns:
            StepResult with the new state, token usage, and the LLM response.
        """
        ...
