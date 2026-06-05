"""Unit tests for Agent service."""

from ai_agent.core.models.llm import LLM, LLMSettings, LLMUsage
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.run import RunResult, RunSettings
from ai_agent.core.models.tool import ToolDefinition
from ai_agent.core.services.agent import Agent


_LLM = LLM(provider="test", model="test-model")
_SETTINGS = LLMSettings(temperature=0.7, max_tokens=4096)
_RUN_SETTINGS = RunSettings(agent="default", llm=_LLM, settings=_SETTINGS)


def _usage() -> LLMUsage:
    return LLMUsage(input_tokens=5, output_tokens=3)


def _run_result(output: str = "reply", messages: list[Message] | None = None) -> RunResult:
    return RunResult(
        output=output,
        turns=1,
        billed_usage=_usage(),
        context_usage=_usage(),
        messages=messages or [],
    )


class _StubRunService:
    """Captures the message list passed to run()."""

    def __init__(self, result: RunResult | None = None) -> None:
        self._result = result or _run_result()
        self.calls: list[list[Message]] = []

    def run(
        self,
        messages: list[Message],
        provider: object,
        model: str,
        settings: LLMSettings,
        tools: list[ToolDefinition] | None,
    ) -> RunResult:
        self.calls.append(list(messages))
        return self._result


def _make_agent(
    name: str = "default",
    description: str = "Test agent.",
    system_prompt: str | None = None,
    run_service: _StubRunService | None = None,
    run_settings: RunSettings = _RUN_SETTINGS,
    base_prompt: str = "",
) -> Agent:
    return Agent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        run_service=run_service or _StubRunService(),  # type: ignore[arg-type]
        run_settings=run_settings,
        base_prompt=base_prompt,
    )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestAgentProperties:
    def test_name_returned(self) -> None:
        assert _make_agent(name="coder").name == "coder"

    def test_description_returned(self) -> None:
        assert _make_agent(description="A coding agent.").description == "A coding agent."

    def test_run_settings_returned(self) -> None:
        assert _make_agent(run_settings=_RUN_SETTINGS).run_settings is _RUN_SETTINGS


# ---------------------------------------------------------------------------
# System prompt merging
# ---------------------------------------------------------------------------


class TestAgentSystemPromptMerge:
    def test_no_base_no_system_prompt_is_empty(self) -> None:
        agent = _make_agent(base_prompt="", system_prompt=None)
        assert agent._system_prompt == ""  # noqa: SLF001

    def test_base_only_used_when_system_prompt_absent(self) -> None:
        agent = _make_agent(base_prompt="base", system_prompt=None)
        assert agent._system_prompt == "base"  # noqa: SLF001

    def test_system_prompt_only_used_when_base_absent(self) -> None:
        agent = _make_agent(base_prompt="", system_prompt="sp")
        assert agent._system_prompt == "sp"  # noqa: SLF001

    def test_both_joined_with_double_newline(self) -> None:
        agent = _make_agent(base_prompt="base", system_prompt="sp")
        assert agent._system_prompt == "base\n\nsp"  # noqa: SLF001

    def test_empty_system_prompt_treated_as_absent(self) -> None:
        agent = _make_agent(base_prompt="base", system_prompt="")
        assert agent._system_prompt == "base"  # noqa: SLF001


# ---------------------------------------------------------------------------
# run() — system message injection
# ---------------------------------------------------------------------------


class TestAgentRun:
    def _messages(self) -> list[Message]:
        return [Message(role=Role.USER, content="hello")]

    def test_system_message_prepended_when_prompt_set(self) -> None:
        svc = _StubRunService()
        agent = _make_agent(system_prompt="be helpful", run_service=svc)
        agent.run(self._messages(), provider=None, model="m", settings=_SETTINGS, tools=None)
        assert svc.calls[0][0].role == Role.SYSTEM

    def test_system_message_content_is_merged_prompt(self) -> None:
        svc = _StubRunService()
        agent = _make_agent(base_prompt="base", system_prompt="sp", run_service=svc)
        agent.run(self._messages(), provider=None, model="m", settings=_SETTINGS, tools=None)
        assert svc.calls[0][0].content == "base\n\nsp"

    def test_caller_messages_follow_system_message(self) -> None:
        svc = _StubRunService()
        agent = _make_agent(system_prompt="be helpful", run_service=svc)
        agent.run(self._messages(), provider=None, model="m", settings=_SETTINGS, tools=None)
        assert svc.calls[0][1].role == Role.USER
        assert svc.calls[0][1].content == "hello"

    def test_total_message_count_is_system_plus_caller(self) -> None:
        svc = _StubRunService()
        agent = _make_agent(system_prompt="be helpful", run_service=svc)
        agent.run(self._messages(), provider=None, model="m", settings=_SETTINGS, tools=None)
        assert len(svc.calls[0]) == 2

    def test_no_system_message_when_prompt_empty(self) -> None:
        svc = _StubRunService()
        agent = _make_agent(base_prompt="", system_prompt=None, run_service=svc)
        agent.run(self._messages(), provider=None, model="m", settings=_SETTINGS, tools=None)
        assert len(svc.calls[0]) == 1
        assert svc.calls[0][0].role == Role.USER

    def test_leading_system_message_merged_not_doubled(self) -> None:
        svc = _StubRunService()
        agent = _make_agent(system_prompt="agent identity", run_service=svc)
        messages = [
            Message(role=Role.SYSTEM, content="compaction summary"),
            Message(role=Role.USER, content="hello"),
        ]
        agent.run(messages, provider=None, model="m", settings=_SETTINGS, tools=None)
        assert svc.calls[0][0].role == Role.SYSTEM
        assert svc.calls[0][1].role == Role.USER

    def test_leading_system_message_content_contains_both(self) -> None:
        svc = _StubRunService()
        agent = _make_agent(system_prompt="agent identity", run_service=svc)
        messages = [
            Message(role=Role.SYSTEM, content="compaction summary"),
            Message(role=Role.USER, content="hello"),
        ]
        agent.run(messages, provider=None, model="m", settings=_SETTINGS, tools=None)
        assert svc.calls[0][0].content == "agent identity\n\ncompaction summary"

    def test_leading_system_message_count_unchanged(self) -> None:
        svc = _StubRunService()
        agent = _make_agent(system_prompt="agent identity", run_service=svc)
        messages = [
            Message(role=Role.SYSTEM, content="compaction summary"),
            Message(role=Role.USER, content="hello"),
        ]
        agent.run(messages, provider=None, model="m", settings=_SETTINGS, tools=None)
        assert len(svc.calls[0]) == 2

    def test_agent_system_stripped_from_result_messages(self) -> None:
        full_trace = [
            Message(role=Role.SYSTEM, content="identity"),
            Message(role=Role.USER, content="hello"),
            Message(role=Role.ASSISTANT, content="reply"),
        ]
        svc = _StubRunService(result=_run_result(messages=full_trace))
        agent = _make_agent(system_prompt="identity", run_service=svc)
        result = agent.run(
            [Message(role=Role.USER, content="hello")],
            provider=None,
            model="m",
            settings=_SETTINGS,
            tools=None,
        )
        assert result.messages[0].role == Role.USER

    def test_compaction_system_restored_in_result_messages(self) -> None:
        compaction_summary = Message(role=Role.SYSTEM, content="summary")
        full_trace = [
            Message(role=Role.SYSTEM, content="identity\n\nsummary"),
            Message(role=Role.USER, content="hello"),
            Message(role=Role.ASSISTANT, content="reply"),
        ]
        svc = _StubRunService(result=_run_result(messages=full_trace))
        agent = _make_agent(system_prompt="identity", run_service=svc)
        result = agent.run(
            [compaction_summary, Message(role=Role.USER, content="hello")],
            provider=None,
            model="m",
            settings=_SETTINGS,
            tools=None,
        )
        assert result.messages[0] == compaction_summary

    def test_result_messages_unchanged_when_no_system_prompt(self) -> None:
        trace = [
            Message(role=Role.USER, content="hello"),
            Message(role=Role.ASSISTANT, content="reply"),
        ]
        svc = _StubRunService(result=_run_result(messages=trace))
        agent = _make_agent(base_prompt="", system_prompt=None, run_service=svc)
        result = agent.run(
            [Message(role=Role.USER, content="hello")],
            provider=None,
            model="m",
            settings=_SETTINGS,
            tools=None,
        )
        assert result.messages == trace

    def test_returns_service_run_result(self) -> None:
        expected = _run_result(output="pong")
        svc = _StubRunService(result=expected)
        agent = _make_agent(run_service=svc)
        result = agent.run(
            self._messages(), provider=None, model="m", settings=_SETTINGS, tools=None
        )
        assert result is expected
