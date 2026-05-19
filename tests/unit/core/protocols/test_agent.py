"""Unit tests for IAgent Protocol."""

from ai_agent.core.models.llm import LLMSettings, LLMUsage
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.run import RunResult, RunSettings
from ai_agent.core.models.llm import LLM
from ai_agent.core.models.agent import Agent
from ai_agent.core.models.tool import ToolDefinition
from ai_agent.core.protocols import IAgent, ILLMProvider


_RUN_SETTINGS = RunSettings(
    agent=Agent(type="node", name="test"),
    llm=LLM(provider="test", model="test-model"),
    settings=LLMSettings(temperature=0.7, max_tokens=4096),
)

_RUN_RESULT = RunResult(
    output="ok",
    turns=1,
    billed_usage=LLMUsage(input_tokens=1, output_tokens=1),
    context_usage=LLMUsage(input_tokens=1, output_tokens=1),
)


class _StubAgent:
    """Minimal concrete class satisfying IAgent."""

    @property
    def run_settings(self) -> RunSettings:
        return _RUN_SETTINGS

    def run(
        self,
        messages: list[Message],
        provider: ILLMProvider,
        model: str,
        settings: LLMSettings,
        tools: list[ToolDefinition] | None,
    ) -> RunResult:
        return _RUN_RESULT


def _accepts_agent(a: IAgent) -> str:
    return "ok"


class TestIAgent:
    """Tests for IAgent Protocol structural subtyping."""

    def test_stub_satisfies_protocol(self) -> None:
        assert _accepts_agent(_StubAgent()) == "ok"

    def test_run_returns_run_result(self) -> None:
        agent = _StubAgent()
        result = agent.run(
            messages=[Message(role=Role.USER, content="hi")],
            provider=None,  # type: ignore[arg-type]
            model="test",
            settings=_RUN_SETTINGS.settings,
            tools=None,
        )
        assert isinstance(result, RunResult)

    def test_run_settings_returns_run_settings(self) -> None:
        agent = _StubAgent()
        assert isinstance(agent.run_settings, RunSettings)

    def test_run_settings_is_correct(self) -> None:
        agent = _StubAgent()
        assert agent.run_settings is _RUN_SETTINGS

    def test_class_missing_run_fails_isinstance(self) -> None:
        class _NoRun:
            @property
            def run_settings(self) -> RunSettings:
                return _RUN_SETTINGS

        assert not isinstance(_NoRun(), IAgent)

    def test_class_missing_run_settings_fails_isinstance(self) -> None:
        class _NoRunSettings:
            def run(
                self,
                messages: list[Message],
                provider: ILLMProvider,
                model: str,
                settings: LLMSettings,
                tools: list[ToolDefinition] | None,
            ) -> RunResult:
                return _RUN_RESULT

        assert not isinstance(_NoRunSettings(), IAgent)
