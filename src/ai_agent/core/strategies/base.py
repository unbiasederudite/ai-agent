"""Base class for all reasoning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

from ai_agent.core.models.agent import AgentState, StepResult
from ai_agent.core.models.llm import LLMRequest
from ai_agent.core.models.strategy import BaseStrategyConfig
from ai_agent.core.protocols.llm import ILLMProvider
from ai_agent.core.services.tool import ToolService


class BaseStrategy(ABC):
    """Shared construction for all reasoning strategy implementations."""

    def __init__(self, config: BaseStrategyConfig, tool_service: ToolService) -> None:
        self._config: Final = config
        self._tool_service: Final = tool_service

    @property
    def config(self) -> BaseStrategyConfig:
        return self._config

    @abstractmethod
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
