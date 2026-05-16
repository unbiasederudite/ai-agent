"""Unit tests for AgentStatus enum and AgentState model and transitions."""

import pytest
from pydantic import ValidationError

from ai_agent.core.models.llm import FinishReason, LLMResponse, LLMUsage
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.agent import AgentState, AgentStatus, StepResult
from ai_agent.core.models.tool import ToolCall


class TestAgentStatus:
    """Tests for the AgentStatus control-signal enum."""

    def test_status_running(self) -> None:
        assert AgentStatus.RUNNING == "running"

    def test_status_awaiting_tools(self) -> None:
        assert AgentStatus.AWAITING_TOOLS == "awaiting_tools"

    def test_status_complete(self) -> None:
        assert AgentStatus.COMPLETE == "complete"

    def test_status_error(self) -> None:
        assert AgentStatus.ERROR == "error"


class TestAgentStateConstruction:
    """Tests for constructing a valid AgentState."""

    def test_agent_state_constructs_with_defaults(self) -> None:
        state = AgentState()
        assert state.status == AgentStatus.RUNNING
        assert state.messages == []
        assert state.turn == 0

    def test_agent_state_has_no_config_field(self) -> None:
        state = AgentState()
        assert not hasattr(state, "config")

    def test_agent_state_is_frozen(self) -> None:
        state = AgentState()
        with pytest.raises(Exception):
            state.status = AgentStatus.COMPLETE  # type: ignore[misc]

    def test_agent_state_stores_messages(self) -> None:
        msg = Message(role=Role.USER, content="hello")
        state = AgentState(messages=[msg])
        assert state.messages[0].content == "hello"

    def test_agent_state_turn_defaults_to_zero(self) -> None:
        state = AgentState()
        assert state.turn == 0

    def test_agent_state_turn_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            AgentState(turn=-1)

    def test_agent_state_has_no_tool_calls_field(self) -> None:
        state = AgentState()
        assert not hasattr(state, "tool_calls")

    def test_agent_state_has_no_tool_results_field(self) -> None:
        state = AgentState()
        assert not hasattr(state, "tool_results")

    def test_agent_state_tool_calls_live_in_messages(self) -> None:
        tc = ToolCall(id="call_1", name="calc", arguments={"x": 1})
        msg = Message(role=Role.ASSISTANT, tool_calls=[tc])
        state = AgentState(messages=[msg])
        assert state.messages[0].tool_calls is not None
        assert state.messages[0].tool_calls[0].id == "call_1"

    def test_agent_state_tool_results_live_in_messages(self) -> None:
        result_msg = Message(role=Role.TOOL, content="42", tool_call_id="call_1")
        state = AgentState(messages=[result_msg])
        assert state.messages[0].role == Role.TOOL
        assert state.messages[0].content == "42"


class TestAgentStateTransitions:
    """Tests for state transition via model_copy."""

    def test_model_copy_increments_turn(self) -> None:
        state = AgentState()
        next_state = state.model_copy(update={"turn": state.turn + 1})
        assert next_state.turn == 1
        assert state.turn == 0  # original unchanged

    def test_model_copy_changes_status(self) -> None:
        state = AgentState()
        next_state = state.model_copy(update={"status": AgentStatus.AWAITING_TOOLS})
        assert next_state.status == AgentStatus.AWAITING_TOOLS
        assert state.status == AgentStatus.RUNNING  # original unchanged

    def test_model_copy_appends_message_immutably(self) -> None:
        state = AgentState()
        msg = Message(role=Role.ASSISTANT, content="hi")
        next_state = state.model_copy(update={"messages": [*state.messages, msg]})
        assert len(next_state.messages) == 1
        assert len(state.messages) == 0  # original unchanged

    def test_model_copy_preserves_unchanged_fields(self) -> None:
        state = AgentState(turn=3)
        next_state = state.model_copy(update={"status": AgentStatus.COMPLETE})
        assert next_state.turn == 3


class TestStepResult:
    """Tests for StepResult — the return type of IReasoningStrategy.step()."""

    def _make_usage(self) -> LLMUsage:
        return LLMUsage(input_tokens=10, output_tokens=5)

    def _make_response(self, content: str = "ok") -> LLMResponse:
        return LLMResponse(
            message=Message(role=Role.ASSISTANT, content=content),
            finish_reason=FinishReason.STOP,
            usage=self._make_usage(),
        )

    def test_constructs_with_required_fields(self) -> None:
        state = AgentState()
        result = StepResult(state=state, response=self._make_response())
        assert result.state is state
        assert result.response.usage.input_tokens == 10
        assert result.response.finish_reason == FinishReason.STOP

    def test_is_frozen(self) -> None:
        result = StepResult(state=AgentState(), response=self._make_response())
        with pytest.raises(Exception):
            result.state = AgentState()  # type: ignore[misc]

    def test_all_fields_are_required(self) -> None:
        with pytest.raises(Exception):
            StepResult()  # type: ignore[call-arg]
