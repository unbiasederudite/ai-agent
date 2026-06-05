"""Unit tests for the Message model."""

import pytest

from ai_agent.core.models.message import Message, Role, append_system, prepend_system, strip_system
from ai_agent.core.models.tool import ToolCall


class TestRole:
    """Tests for the Role enum."""

    def test_role_has_system(self) -> None:
        assert Role.SYSTEM == "system"

    def test_role_has_user(self) -> None:
        assert Role.USER == "user"

    def test_role_has_assistant(self) -> None:
        assert Role.ASSISTANT == "assistant"

    def test_role_has_tool(self) -> None:
        assert Role.TOOL == "tool"


class TestMessage:
    """Tests for the Message model."""

    def test_message_constructs_with_role_and_content(self) -> None:
        msg = Message(role=Role.USER, content="hello")
        assert msg.role == Role.USER
        assert msg.content == "hello"

    def test_message_is_frozen(self) -> None:
        msg = Message(role=Role.USER, content="hello")
        with pytest.raises(Exception):
            msg.content = "modified"  # type: ignore[misc]

    def test_message_content_is_optional(self) -> None:
        # Assistant messages that only issue tool calls may have no text content.
        msg = Message(role=Role.ASSISTANT)
        assert msg.content is None

    def test_message_tool_call_id_is_optional(self) -> None:
        # tool_call_id is None by default on non-TOOL roles
        msg = Message(role=Role.ASSISTANT, content="reply")
        assert msg.tool_call_id is None

    def test_message_tool_call_id_can_be_set(self) -> None:
        msg = Message(role=Role.TOOL, content="result", tool_call_id="call_abc123")
        assert msg.tool_call_id == "call_abc123"

    def test_message_system_role_accepted(self) -> None:
        msg = Message(role=Role.SYSTEM, content="You are an assistant.")
        assert msg.role == Role.SYSTEM

    def test_message_invalid_role_raises(self) -> None:
        with pytest.raises(Exception):
            Message(role="invalid_role", content="hi")  # type: ignore[arg-type]

    def test_message_tool_calls_is_optional(self) -> None:
        msg = Message(role=Role.ASSISTANT, content="thinking...")
        assert msg.tool_calls is None

    def test_message_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(id="call_1", name="calculator", arguments={"x": 1})
        msg = Message(role=Role.ASSISTANT, tool_calls=[tc])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_1"
        assert msg.content is None  # no text content — only tool calls

    def test_message_tool_result_round_trip(self) -> None:
        msg = Message(
            role=Role.TOOL,
            content="42",
            tool_call_id="call_1",
        )
        assert msg.role == Role.TOOL
        assert msg.content == "42"
        assert msg.tool_call_id == "call_1"


def _sys(content: str) -> Message:
    return Message(role=Role.SYSTEM, content=content)


def _user(content: str = "hello") -> Message:
    return Message(role=Role.USER, content=content)


class TestStripSystem:
    def test_removes_leading_system(self) -> None:
        result = strip_system([_sys("s"), _user()])
        assert result[0].role == Role.USER

    def test_returns_tail_unchanged(self) -> None:
        user = _user()
        result = strip_system([_sys("s"), user])
        assert result[0] is user

    def test_no_system_returns_list_unchanged(self) -> None:
        msgs = [_user(), _user("bye")]
        assert strip_system(msgs) == msgs

    def test_empty_list_returns_empty(self) -> None:
        assert strip_system([]) == []

    def test_returns_new_list(self) -> None:
        msgs = [_user()]
        assert strip_system(msgs) is not msgs


class TestPrependSystem:
    def test_creates_system_when_absent(self) -> None:
        result = prepend_system([_user()], "identity")
        assert result[0].role == Role.SYSTEM
        assert result[0].content == "identity"

    def test_user_message_follows_created_system(self) -> None:
        result = prepend_system([_user()], "identity")
        assert result[1].role == Role.USER

    def test_merges_before_existing_system(self) -> None:
        result = prepend_system([_sys("summary"), _user()], "identity")
        assert result[0].content == "identity\n\nsummary"

    def test_existing_system_only_one_after_merge(self) -> None:
        result = prepend_system([_sys("summary"), _user()], "identity")
        assert sum(1 for m in result if m.role == Role.SYSTEM) == 1

    def test_empty_existing_content_uses_new_content_only(self) -> None:
        result = prepend_system([_sys(""), _user()], "identity")
        assert result[0].content == "identity"

    def test_none_existing_content_uses_new_content_only(self) -> None:
        sys_msg = Message(role=Role.SYSTEM, content=None)
        result = prepend_system([sys_msg, _user()], "identity")
        assert result[0].content == "identity"


class TestAppendSystem:
    def test_creates_system_when_absent(self) -> None:
        result = append_system([_user()], "strategy")
        assert result[0].role == Role.SYSTEM
        assert result[0].content == "strategy"

    def test_user_message_follows_created_system(self) -> None:
        result = append_system([_user()], "strategy")
        assert result[1].role == Role.USER

    def test_merges_after_existing_system(self) -> None:
        result = append_system([_sys("identity"), _user()], "strategy")
        assert result[0].content == "identity\n\nstrategy"

    def test_existing_system_only_one_after_merge(self) -> None:
        result = append_system([_sys("identity"), _user()], "strategy")
        assert sum(1 for m in result if m.role == Role.SYSTEM) == 1

    def test_empty_existing_content_uses_new_content_only(self) -> None:
        result = append_system([_sys(""), _user()], "strategy")
        assert result[0].content == "strategy"

    def test_none_existing_content_uses_new_content_only(self) -> None:
        sys_msg = Message(role=Role.SYSTEM, content=None)
        result = append_system([sys_msg, _user()], "strategy")
        assert result[0].content == "strategy"
