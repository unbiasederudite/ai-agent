"""Unit tests for IReasoningStrategy Protocol."""

from ai_agent.core.models.llm import FinishReason, LLMRequest, LLMResponse, LLMSettings, LLMUsage
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.agent import AgentState, AgentStatus, StepResult
from ai_agent.core.models.strategy import BaseStrategyConfig
from ai_agent.core.protocols import ILLMProvider, IReasoningStrategy
from ai_agent.core.registries.tool import ToolRegistry
from ai_agent.core.services.tool import ToolService
from ai_agent.core.strategies.base import BaseStrategy


_CONFIG = BaseStrategyConfig(type="stub")


def _tool_service() -> ToolService:
    return ToolService(registry=ToolRegistry())


def _make_response() -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content="ok"),
        finish_reason=FinishReason.STOP,
        usage=LLMUsage(input_tokens=1, output_tokens=1),
    )


class _StubStrategy(BaseStrategy):
    """Minimal concrete class satisfying IReasoningStrategy."""

    def step(self, state: AgentState, request: LLMRequest, provider: ILLMProvider) -> StepResult:
        new_state = state.model_copy(update={"status": AgentStatus.COMPLETE})
        return StepResult(state=new_state, response=_make_response())


def _stub() -> _StubStrategy:
    return _StubStrategy(_CONFIG, _tool_service())


def _accepts_strategy(s: IReasoningStrategy) -> str:
    return "ok"


_REQUEST = LLMRequest(
    model="test",
    settings=LLMSettings(temperature=0.7, max_tokens=1024),
    messages=[Message(role=Role.USER, content="hi")],
)


class TestIReasoningStrategy:
    """Tests for IReasoningStrategy Protocol structural subtyping."""

    def test_stub_satisfies_protocol(self) -> None:
        assert _accepts_strategy(_stub()) == "ok"

    def test_base_strategy_subclass_satisfies_protocol(self) -> None:
        assert isinstance(_stub(), IReasoningStrategy)

    def test_step_returns_step_result(self) -> None:
        result = _stub().step(AgentState(), _REQUEST, provider=None)  # type: ignore[arg-type]
        assert isinstance(result, StepResult)

    def test_step_result_state_is_agent_state(self) -> None:
        result = _stub().step(AgentState(), _REQUEST, provider=None)  # type: ignore[arg-type]
        assert isinstance(result.state, AgentState)

    def test_step_result_response_is_llm_response(self) -> None:
        result = _stub().step(AgentState(), _REQUEST, provider=None)  # type: ignore[arg-type]
        assert isinstance(result.response, LLMResponse)

    def test_config_is_strategy_config(self) -> None:
        assert isinstance(_stub().config, BaseStrategyConfig)

    def test_config_is_the_injected_instance(self) -> None:
        assert _stub().config is _CONFIG

    def test_class_missing_step_fails_isinstance(self) -> None:
        class _NoStep:
            @property
            def config(self) -> BaseStrategyConfig:
                return _CONFIG

        assert not isinstance(_NoStep(), IReasoningStrategy)

    def test_class_missing_config_fails_isinstance(self) -> None:
        class _NoConfig:
            def step(
                self, state: AgentState, request: LLMRequest, provider: ILLMProvider
            ) -> StepResult: ...  # type: ignore[return-value]

        assert not isinstance(_NoConfig(), IReasoningStrategy)
