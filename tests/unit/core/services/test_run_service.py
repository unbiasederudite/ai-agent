"""Unit tests for RunService.

RunService is a pure loop runner — it only understands COMPLETE and ERROR.
All other state transitions (tool dispatch, retries) are the strategy's concern.
"""

import pytest

from ai_agent.core.exceptions import LoopDetectedError, ReasoningError
from ai_agent.core.models.llm import FinishReason, LLMRequest, LLMResponse, LLMSettings, LLMUsage
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.run import RunResult
from ai_agent.core.models.agent import AgentState, AgentStatus, StepResult
from ai_agent.core.models.strategy import StrategyConfig
from ai_agent.core.models.tool import ToolDefinition
from ai_agent.core.services import RunService


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_usage(inp: int = 5, out: int = 3) -> LLMUsage:
    return LLMUsage(input_tokens=inp, output_tokens=out)


def _make_response(
    content: str | None = "done",
    finish_reason: FinishReason = FinishReason.STOP,
) -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=content),
        finish_reason=finish_reason,
        usage=_make_usage(),
    )


class _StubProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        return _make_response()

    def context_window(self, model: str) -> int:
        return 128_000


_PROVIDER = _StubProvider()
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=4096)
_MODEL = "mock-model"


# ---------------------------------------------------------------------------
# Mock strategies
# ---------------------------------------------------------------------------


class _CompleteOnFirstStep:
    config: StrategyConfig = StrategyConfig(type="stub")

    def step(self, state: AgentState, request: LLMRequest, provider: object = None) -> StepResult:
        msg = Message(role=Role.ASSISTANT, content="done")
        new_state = state.model_copy(
            update={"status": AgentStatus.COMPLETE, "messages": [*state.messages, msg]}
        )
        return StepResult(state=new_state, response=_make_response("done"))


class _ErrorOnFirstStep:
    config: StrategyConfig = StrategyConfig(type="stub")

    def step(self, state: AgentState, request: LLMRequest, provider: object = None) -> StepResult:
        new_state = state.model_copy(update={"status": AgentStatus.ERROR})
        return StepResult(state=new_state, response=_make_response())


class _RunningThenComplete:
    """Runs one RUNNING step then completes — simulates a multi-turn strategy."""

    config: StrategyConfig = StrategyConfig(type="stub")

    def __init__(self) -> None:
        self._calls = 0

    def step(self, state: AgentState, request: LLMRequest, provider: object = None) -> StepResult:
        self._calls += 1
        if self._calls == 1:
            new_state = state.model_copy(update={"status": AgentStatus.RUNNING})
            return StepResult(state=new_state, response=_make_response("partial"))
        msg = Message(role=Role.ASSISTANT, content="done")
        new_state = state.model_copy(
            update={"status": AgentStatus.COMPLETE, "messages": [*state.messages, msg]}
        )
        return StepResult(state=new_state, response=_make_response("done"))


class _NeverCompletes:
    config: StrategyConfig = StrategyConfig(type="stub", max_turns=3)

    def step(self, state: AgentState, request: LLMRequest, provider: object = None) -> StepResult:
        new_state = state.model_copy(update={"status": AgentStatus.RUNNING})
        return StepResult(state=new_state, response=_make_response())


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def _make_service(strategy: object | None = None) -> RunService:
    return RunService(strategy=strategy or _CompleteOnFirstStep())  # type: ignore[arg-type]


def _run(
    svc: RunService,
    messages: list[Message] | None = None,
    provider: object | None = None,
    model: str = _MODEL,
    settings: LLMSettings | None = None,
    tools: list[ToolDefinition] | None = None,
) -> RunResult:
    return svc.run(
        messages=messages or [Message(role=Role.USER, content="hello")],
        provider=provider or _PROVIDER,  # type: ignore[arg-type]
        model=model,
        settings=settings or _SETTINGS,
        tools=tools,
    )


# ---------------------------------------------------------------------------
# RunService — basic loop
# ---------------------------------------------------------------------------


class TestRunServiceLoop:
    def test_returns_run_result(self) -> None:
        svc = _make_service()
        assert isinstance(_run(svc), RunResult)

    def test_output_is_last_assistant_content(self) -> None:
        svc = _make_service()
        assert _run(svc).output == "done"

    def test_error_status_raises_reasoning_error(self) -> None:
        svc = _make_service(strategy=_ErrorOnFirstStep())
        with pytest.raises(ReasoningError):
            _run(svc)

    def test_running_status_continues_loop(self) -> None:
        strategy = _RunningThenComplete()
        svc = _make_service(strategy=strategy)
        result = _run(svc)
        assert result.output == "done"
        assert strategy._calls == 2


# ---------------------------------------------------------------------------
# RunService — loop detection
# ---------------------------------------------------------------------------


class TestRunServiceLoopDetection:
    def test_exceeding_max_turns_raises_loop_detected(self) -> None:
        svc = _make_service(strategy=_NeverCompletes())
        with pytest.raises(LoopDetectedError):
            _run(svc)

    def test_completes_within_max_turns_succeeds(self) -> None:
        class _CompleteWithTightLimit:
            config: StrategyConfig = StrategyConfig(type="stub", max_turns=1)

            def step(
                self, state: AgentState, request: LLMRequest, provider: object = None
            ) -> StepResult:
                msg = Message(role=Role.ASSISTANT, content="done")
                new_state = state.model_copy(
                    update={"status": AgentStatus.COMPLETE, "messages": [*state.messages, msg]}
                )
                return StepResult(state=new_state, response=_make_response("done"))

        svc = _make_service(strategy=_CompleteWithTightLimit())
        result = _run(svc)
        assert result.output == "done"


# ---------------------------------------------------------------------------
# RunService — no assistant content raises
# ---------------------------------------------------------------------------


class TestRunServiceNoOutput:
    def test_no_assistant_content_raises_reasoning_error(self) -> None:
        class _NoContentStrategy:
            config: StrategyConfig = StrategyConfig(type="stub")

            def step(
                self, state: AgentState, request: LLMRequest, provider: object = None
            ) -> StepResult:
                msg = Message(role=Role.ASSISTANT, content=None)
                new_state = state.model_copy(
                    update={"status": AgentStatus.COMPLETE, "messages": [*state.messages, msg]}
                )
                return StepResult(state=new_state, response=_make_response(content=None))

        svc = _make_service(strategy=_NoContentStrategy())
        with pytest.raises(ReasoningError):
            _run(svc)


# ---------------------------------------------------------------------------
# RunService — token usage accumulation
# ---------------------------------------------------------------------------


class TestRunServiceUsage:
    def test_run_result_has_billed_usage(self) -> None:
        svc = _make_service()
        result = _run(svc)
        assert isinstance(result.billed_usage, LLMUsage)

    def test_run_result_has_context_usage(self) -> None:
        svc = _make_service()
        result = _run(svc)
        assert isinstance(result.context_usage, LLMUsage)

    def test_single_step_billed_usage(self) -> None:
        svc = _make_service()
        result = _run(svc)
        assert result.billed_usage.input_tokens == 5
        assert result.billed_usage.output_tokens == 3

    def test_single_step_context_usage_equals_billed(self) -> None:
        svc = _make_service()
        result = _run(svc)
        assert result.context_usage.input_tokens == 5
        assert result.context_usage.output_tokens == 3

    def test_multi_step_billed_usage_accumulated(self) -> None:
        svc = _make_service(strategy=_RunningThenComplete())
        result = _run(svc)
        assert result.billed_usage.input_tokens == 10
        assert result.billed_usage.output_tokens == 6

    def test_multi_step_context_usage_is_last_step_only(self) -> None:
        svc = _make_service(strategy=_RunningThenComplete())
        result = _run(svc)
        assert result.context_usage.input_tokens == 5
        assert result.context_usage.output_tokens == 3
