"""Unit tests for IReasoningStrategy Protocol."""

from ai_agent.core.models.llm import FinishReason, LLMRequest, LLMResponse, LLMSettings, LLMUsage
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.agent import AgentState, AgentStatus, StepResult
from ai_agent.core.models.strategy import StrategyConfig
from ai_agent.core.protocols import ILLMProvider, IReasoningStrategy


def _make_response() -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content="ok"),
        finish_reason=FinishReason.STOP,
        usage=LLMUsage(input_tokens=1, output_tokens=1),
    )


class _StubStrategy:
    """Minimal concrete class satisfying IReasoningStrategy."""

    config: StrategyConfig = StrategyConfig(type="stub")

    def step(self, state: AgentState, request: LLMRequest, provider: ILLMProvider) -> StepResult:
        new_state = state.model_copy(update={"status": AgentStatus.COMPLETE})
        return StepResult(state=new_state, response=_make_response())


def _accepts_strategy(s: IReasoningStrategy) -> str:
    return "ok"


class TestIReasoningStrategy:
    """Tests for IReasoningStrategy Protocol structural subtyping."""

    def test_stub_satisfies_protocol(self) -> None:
        assert _accepts_strategy(_StubStrategy()) == "ok"

    def test_step_returns_step_result(self) -> None:
        strategy = _StubStrategy()
        state = AgentState()
        request = LLMRequest(
            model="test",
            settings=LLMSettings(temperature=0.7, max_tokens=1024),
            messages=[Message(role=Role.USER, content="hi")],
        )
        result = strategy.step(state, request, provider=None)  # type: ignore[arg-type]
        assert isinstance(result, StepResult)

    def test_step_result_state_is_agent_state(self) -> None:
        strategy = _StubStrategy()
        state = AgentState()
        request = LLMRequest(
            model="test",
            settings=LLMSettings(temperature=0.7, max_tokens=1024),
            messages=[Message(role=Role.USER, content="hi")],
        )
        result = strategy.step(state, request, provider=None)  # type: ignore[arg-type]
        assert isinstance(result.state, AgentState)

    def test_step_result_response_is_llm_response(self) -> None:
        strategy = _StubStrategy()
        state = AgentState()
        request = LLMRequest(
            model="test",
            settings=LLMSettings(temperature=0.7, max_tokens=1024),
            messages=[Message(role=Role.USER, content="hi")],
        )
        result = strategy.step(state, request, provider=None)  # type: ignore[arg-type]
        assert isinstance(result.response, LLMResponse)

    def test_config_is_strategy_config(self) -> None:
        assert isinstance(_StubStrategy().config, StrategyConfig)

    def test_class_missing_step_fails_isinstance(self) -> None:
        class _NoStep:
            config: StrategyConfig = StrategyConfig(type="stub")

        assert not isinstance(_NoStep(), IReasoningStrategy)

    def test_class_missing_config_fails_isinstance(self) -> None:
        class _NoConfig:
            def step(
                self, state: AgentState, request: LLMRequest, provider: ILLMProvider
            ) -> StepResult: ...  # type: ignore[return-value]

        assert not isinstance(_NoConfig(), IReasoningStrategy)
