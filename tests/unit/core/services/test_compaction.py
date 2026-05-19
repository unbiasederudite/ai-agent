"""Unit tests for CompactionService."""

import pytest

from ai_agent.core.exceptions import CompletionError
from ai_agent.core.services.compaction import CompactionService
from ai_agent.core.models import (
    CompactionResult,
    FinishReason,
    LLMRequest,
    LLMResponse,
    LLMSettings,
    LLMUsage,
    Message,
    Role,
)


def _user(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def _assistant(content: str) -> Message:
    return Message(role=Role.ASSISTANT, content=content)


def _system(content: str) -> Message:
    return Message(role=Role.SYSTEM, content=content)


def _tool(content: str, call_id: str = "c1") -> Message:
    return Message(role=Role.TOOL, content=content, tool_call_id=call_id)


class _StubProvider:
    def __init__(self, summary: str = "summary text") -> None:
        self._summary = summary
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            message=Message(role=Role.ASSISTANT, content=self._summary),
            finish_reason=FinishReason.STOP,
            usage=LLMUsage(input_tokens=20, output_tokens=8),
        )

    def context_window(self, model: str) -> int:
        return 128_000


def _make_service(
    provider: _StubProvider | None = None,
    max_tokens: int = 512,
) -> CompactionService:
    return CompactionService(
        provider=provider or _StubProvider(),
        model="mock-model",
        settings=LLMSettings(temperature=0.7, max_tokens=max_tokens),
    )


# ---------------------------------------------------------------------------
# _split_into_turns
# ---------------------------------------------------------------------------


class TestSplitIntoTurns:
    def test_empty_produces_no_turns(self) -> None:
        assert CompactionService._split_into_turns([]) == []

    def test_single_user_message_is_one_turn(self) -> None:
        assert len(CompactionService._split_into_turns([_user("hi")])) == 1

    def test_user_assistant_is_one_turn(self) -> None:
        turns = CompactionService._split_into_turns([_user("q"), _assistant("a")])
        assert len(turns) == 1
        assert len(turns[0]) == 2

    def test_two_user_messages_produce_two_turns(self) -> None:
        turns = CompactionService._split_into_turns(
            [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
        )
        assert len(turns) == 2

    def test_tool_message_stays_in_same_turn(self) -> None:
        turns = CompactionService._split_into_turns([_user("q"), _tool("result"), _assistant("a")])
        assert len(turns) == 1
        assert len(turns[0]) == 3

    def test_incomplete_final_turn_included(self) -> None:
        turns = CompactionService._split_into_turns([_user("q1"), _assistant("a1"), _user("q2")])
        assert len(turns) == 2
        assert turns[-1] == [_user("q2")]


# ---------------------------------------------------------------------------
# compact — no-op cases
# ---------------------------------------------------------------------------


class TestCompactNoOp:
    def test_returns_unchanged_when_turns_equal_keep(self) -> None:
        msgs = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
        result = _make_service().compact(msgs, keep_recent_turns=2)
        assert result.messages == msgs

    def test_returns_unchanged_when_turns_fewer_than_keep(self) -> None:
        msgs = [_user("q"), _assistant("a")]
        result = _make_service().compact(msgs, keep_recent_turns=2)
        assert result.messages == msgs

    def test_empty_messages_returns_unchanged(self) -> None:
        result = _make_service().compact([], keep_recent_turns=2)
        assert result.messages == []

    def test_no_op_usage_is_none(self) -> None:
        result = _make_service().compact([_user("q"), _assistant("a")], keep_recent_turns=2)
        assert result.usage is None

    def test_no_llm_call_when_no_compaction(self) -> None:
        provider = _StubProvider()
        msgs = [_user("q"), _assistant("a")]
        _make_service(provider).compact(msgs, keep_recent_turns=2)
        assert len(provider.requests) == 0


# ---------------------------------------------------------------------------
# compact — output structure
# ---------------------------------------------------------------------------


class TestCompactOutput:
    def _three_turn_msgs(self) -> list[Message]:
        return [
            _user("q1"),
            _assistant("a1"),
            _user("q2"),
            _assistant("a2"),
            _user("q3"),
            _assistant("a3"),
        ]

    def test_returns_compaction_result(self) -> None:
        result = _make_service().compact(self._three_turn_msgs(), keep_recent_turns=2)
        assert isinstance(result, CompactionResult)

    def test_first_message_is_system_summary(self) -> None:
        result = _make_service().compact(self._three_turn_msgs(), keep_recent_turns=2)
        assert result.messages[0].role == Role.SYSTEM

    def test_summary_content_is_provider_reply(self) -> None:
        provider = _StubProvider(summary="the summary")
        result = _make_service(provider).compact(self._three_turn_msgs(), keep_recent_turns=2)
        assert result.messages[0].content == "the summary"

    def test_kept_turns_follow_summary(self) -> None:
        result = _make_service().compact(self._three_turn_msgs(), keep_recent_turns=2)
        kept = result.messages[1:]
        assert kept[0].content == "q2"
        assert kept[1].content == "a2"
        assert kept[2].content == "q3"
        assert kept[3].content == "a3"

    def test_total_length_is_summary_plus_kept(self) -> None:
        result = _make_service().compact(self._three_turn_msgs(), keep_recent_turns=2)
        assert len(result.messages) == 1 + 4  # summary + 2 turns × 2 messages

    def test_usage_reflects_llm_call(self) -> None:
        result = _make_service().compact(self._three_turn_msgs(), keep_recent_turns=2)
        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 8

    def test_does_not_mutate_input(self) -> None:
        msgs = self._three_turn_msgs()
        original = list(msgs)
        _make_service().compact(msgs, keep_recent_turns=2)
        assert msgs == original


# ---------------------------------------------------------------------------
# compact — rolling summary
# ---------------------------------------------------------------------------


class TestCompactRollingSummary:
    def test_existing_summary_included_in_llm_input(self) -> None:
        provider = _StubProvider()
        msgs = [
            _system("previous summary"),
            _user("q1"),
            _assistant("a1"),
            _user("q2"),
            _assistant("a2"),
            _user("q3"),
            _assistant("a3"),
        ]
        _make_service(provider).compact(msgs, keep_recent_turns=2)
        sent = provider.requests[0].messages
        assert sent[0].role == Role.SYSTEM
        assert sent[0].content == "previous summary"

    def test_summarised_turns_sent_to_provider(self) -> None:
        provider = _StubProvider()
        msgs = [
            _user("q1"),
            _assistant("a1"),
            _user("q2"),
            _assistant("a2"),
            _user("q3"),
            _assistant("a3"),
        ]
        _make_service(provider).compact(msgs, keep_recent_turns=2)
        sent = provider.requests[0].messages
        contents = [m.content for m in sent if m.role != Role.USER or m.content != sent[-1].content]
        assert "q1" in contents
        assert "a1" in contents

    def test_compaction_prompt_appended_as_user_message(self) -> None:
        provider = _StubProvider()
        msgs = [
            _user("q1"),
            _assistant("a1"),
            _user("q2"),
            _assistant("a2"),
            _user("q3"),
            _assistant("a3"),
        ]
        _make_service(provider).compact(msgs, keep_recent_turns=2)
        last = provider.requests[0].messages[-1]
        assert last.role == Role.USER
        assert len(last.content or "") > 0


# ---------------------------------------------------------------------------
# compact — LLM call parameters
# ---------------------------------------------------------------------------


class TestCompactLLMParams:
    def _msgs(self) -> list[Message]:
        return [
            _user("q1"),
            _assistant("a1"),
            _user("q2"),
            _assistant("a2"),
            _user("q3"),
            _assistant("a3"),
        ]

    def test_uses_configured_model(self) -> None:
        provider = _StubProvider()
        CompactionService(
            provider=provider,
            model="gpt-4o",
            settings=LLMSettings(temperature=0.7, max_tokens=512),
        ).compact(self._msgs(), keep_recent_turns=2)
        assert provider.requests[0].model == "gpt-4o"

    def test_uses_configured_max_tokens(self) -> None:
        provider = _StubProvider()
        _make_service(provider, max_tokens=768).compact(self._msgs(), keep_recent_turns=2)
        assert provider.requests[0].settings.max_tokens == 768

    def test_raises_completion_error_on_empty_summary(self) -> None:
        provider = _StubProvider(summary="")
        with pytest.raises(CompletionError):
            _make_service(provider).compact(self._msgs(), keep_recent_turns=2)
