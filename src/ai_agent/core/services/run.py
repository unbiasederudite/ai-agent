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
    """Executes the agent reasoning loop until a terminal state is reached."""

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
            messages: Message history.
            provider: LLM provider.
            model: Model identifier.
            settings: Sampling parameters.
            tools: Tool definitions, or None.

        Returns:
            Run result.

        Raises:
            ReasoningError: Strategy terminates with ERROR or produces no text reply.
            LoopDetectedError: strategy.config.max_turns exceeded without a terminal status.
        """
        max_turns = self._strategy.config.max_turns
        current = AgentState(messages=messages)
        billed_input_tokens = 0
        billed_output_tokens = 0
        output: str | None = None
        last_usage = LLMUsage(input_tokens=0, output_tokens=0)

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
            billed_input_tokens += result.response.usage.input_tokens
            billed_output_tokens += result.response.usage.output_tokens
            last_usage = result.response.usage

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
            billed_usage=LLMUsage(
                input_tokens=billed_input_tokens, output_tokens=billed_output_tokens
            ),
            context_usage=last_usage,
        )
