"""ReAct (Reason + Act) reasoning strategy."""

from __future__ import annotations

import json
from typing import Final

from ai_agent.core.models.agent import AgentState, AgentStatus, StepResult
from ai_agent.core.models.llm import LLMRequest
from ai_agent.core.models.message import Message, Role, append_system
from ai_agent.core.models.strategy import ReActStrategyConfig
from ai_agent.core.protocols.llm import ILLMProvider
from ai_agent.core.services.tool import ToolService
from ai_agent.core.strategies.base import BaseStrategy

_STRATEGY_PROMPT = """
When you need information or need to take an action, use the available tools.
When you have enough information to answer, respond directly without using tools.
"""


class ReActStrategy(BaseStrategy):
    """Single-step ReAct: one LLM call, then either dispatch tools or terminate."""

    def __init__(
        self,
        config: ReActStrategyConfig,
        tool_service: ToolService,
        strategy_prompt: str = _STRATEGY_PROMPT,
    ) -> None:
        super().__init__(config, tool_service)
        self._strategy_prompt: Final = strategy_prompt

    def step(
        self,
        state: AgentState,
        request: LLMRequest,
        provider: ILLMProvider,
    ) -> StepResult:
        messages = append_system(list(request.messages), self._strategy_prompt)
        request = request.model_copy(update={"messages": messages})
        response = provider.complete(request)

        if response.tool_calls:
            tool_results = self._tool_service.dispatch(response.tool_calls)
            tool_messages = [
                Message(
                    role=Role.TOOL,
                    content=r.content if isinstance(r.content, str) else json.dumps(r.content),
                    tool_call_id=r.id,
                )
                for r in tool_results
            ]
            return StepResult(
                state=AgentState(
                    status=AgentStatus.RUNNING,
                    turn=state.turn + 1,
                    messages=[*state.messages, response.message, *tool_messages],
                ),
                response=response,
            )

        return StepResult(
            state=AgentState(
                status=AgentStatus.COMPLETE,
                turn=state.turn + 1,
                messages=[*state.messages, response.message],
            ),
            response=response,
        )
