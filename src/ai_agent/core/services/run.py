"""Pure reasoning loop execution engine."""

import logging

from ai_agent.core.exceptions import LoopDetectedError, ReasoningError
from ai_agent.core.models.llm import LLMRequest, LLMSettings, LLMUsage
from ai_agent.core.models.message import Message
from ai_agent.core.models.run import RunResult
from ai_agent.core.models.agent import AgentState, AgentStatus
from ai_agent.core.models.tool import ToolDefinition
from ai_agent.core.protocols.llm import ILLMProvider
from ai_agent.core.protocols.strategy import IReasoningStrategy

_log = logging.getLogger(__name__)


class RunService:
    """Executes the agent reasoning loop until a terminal state is reached.

    RunService is a pure loop runner. It only checks for COMPLETE or ERROR.
    All other state transitions (tool dispatch, retries, etc.) are the strategy's concern.
    """

    def __init__(self, strategy: IReasoningStrategy) -> None:
        self._strategy = strategy

    def run(
        self,
        messages: list[Message],
        provider: ILLMProvider,
        model: str,
        settings: LLMSettings,
        tools: list[ToolDefinition] | None,
    ) -> RunResult:
        """Execute the reasoning loop until a terminal state is reached.

        Args:
            messages: Conversation history for this run (system + prior turns + user message).
            provider: Resolved LLM provider.
            model: Model identifier.
            settings: Sampling parameters.
            tools: Tool definitions exposed to the LLM. None disables tool calling.

        Returns:
            RunResult with the assistant reply, turn count, and accumulated token usage.

        Raises:
            ReasoningError: Strategy terminates with ERROR or produces no text reply.
            LoopDetectedError: strategy.config.max_turns exceeded without a terminal status.
        """
        max_turns = self._strategy.config.max_turns
        current = AgentState(messages=messages)
        input_tokens = 0
        output_tokens = 0
        output: str | None = None

        while True:
            if current.turn >= max_turns:
                raise LoopDetectedError(f"Agent exceeded max_turns={max_turns} without completing.")
            _log.debug("run_service.step", extra={"turn": current.turn})
            request = LLMRequest(
                model=model,
                settings=settings,
                messages=current.messages,
                tools=tools,
            )
            result = self._strategy.step(current, request, provider)
            input_tokens += result.response.usage.input_tokens
            output_tokens += result.response.usage.output_tokens

            if result.state.status == AgentStatus.COMPLETE:
                output = result.response.message.content
                current = result.state.model_copy(update={"turn": result.state.turn + 1})
                break

            if result.state.status == AgentStatus.ERROR:
                raise ReasoningError("Agent run terminated with ERROR status.")

            current = result.state.model_copy(update={"turn": result.state.turn + 1})

        if output is None:
            raise ReasoningError("Run completed but no assistant response was produced.")

        return RunResult(
            output=output,
            turns=current.turn,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )
