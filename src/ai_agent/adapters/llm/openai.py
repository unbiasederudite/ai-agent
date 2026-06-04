"""OpenAI LLM provider adapter."""

from __future__ import annotations

import json
import logging
from typing import Any, Final

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    Function as ToolCallFunction,
)
from openai.types.shared_params import FunctionDefinition

from ai_agent.core.exceptions import (
    AuthenticationError,
    CompletionError,
    ContextWindowExceededError,
    RateLimitError,
)
from ai_agent.core.models.llm import FinishReason, LLMRequest, LLMResponse, LLMUsage
from ai_agent.core.models.message import Message, Role
from ai_agent.core.models.tool import ToolCall, ToolDefinition

_logger = logging.getLogger(__name__)

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,  # legacy alias
    "length": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,  # older models
    "content_filter": FinishReason.STOP,
}


class OpenAIAdapter:
    """ILLMProvider implementation backed by the OpenAI chat completions API."""

    def __init__(self, client: openai.OpenAI) -> None:
        """Initialize the adapter with an OpenAI client.

        Args:
            client: Pre-configured OpenAI client instance.
        """
        self._client: Final = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to OpenAI.

        Args:
            request: The request including model, messages, and optional tools.

        Returns:
            A validated LLMResponse.

        Raises:
            AuthenticationError: Invalid API credentials.
            RateLimitError: Request throttled by the provider.
            ContextWindowExceededError: Message list exceeds context window.
            CompletionError: Any other provider failure.
        """
        messages = [self._map_message(m) for m in request.messages]
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.settings.temperature,
            "max_tokens": request.settings.max_tokens,
        }
        if request.tools:
            kwargs["tools"] = [self._map_tool(t) for t in request.tools]

        try:
            raw = self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise AuthenticationError(str(exc)) from exc
        except openai.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise CompletionError(f"Connection failed: {exc}") from exc
        except openai.BadRequestError as exc:
            if exc.code == "context_length_exceeded":
                raise ContextWindowExceededError(str(exc)) from exc
            raise CompletionError(str(exc)) from exc
        except openai.APIError as exc:
            raise CompletionError(str(exc)) from exc

        return self._map_response(raw)

    def _map_message(self, message: Message) -> ChatCompletionMessageParam:
        match message.role:
            case Role.SYSTEM:
                return ChatCompletionSystemMessageParam(
                    role="system", content=message.content or ""
                )
            case Role.USER:
                return ChatCompletionUserMessageParam(role="user", content=message.content or "")
            case Role.TOOL:
                return ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=message.tool_call_id or "",
                    content=message.content or "",
                )
            case Role.ASSISTANT:
                asst: ChatCompletionAssistantMessageParam = {"role": "assistant"}
                if message.content is not None:
                    asst["content"] = message.content
                if message.tool_calls:
                    asst["tool_calls"] = [
                        ChatCompletionMessageToolCallParam(
                            id=tc.id,
                            type="function",
                            function=ToolCallFunction(
                                arguments=json.dumps(tc.arguments), name=tc.name
                            ),
                        )
                        for tc in message.tool_calls
                    ]
                return asst
            case _:
                raise CompletionError(f"Unhandled message role: {message.role!r}")

    def _map_tool(self, tool: ToolDefinition) -> ChatCompletionFunctionToolParam:
        return ChatCompletionFunctionToolParam(
            type="function",
            function=FunctionDefinition(
                name=tool.name,
                description=tool.tool_schema.description,
                parameters=tool.tool_schema.parameters,
            ),
        )

    def _map_response(self, response: ChatCompletion) -> LLMResponse:
        choice = response.choices[0]
        finish_reason = _FINISH_REASONS.get(choice.finish_reason or "", FinishReason.STOP)

        msg = choice.message
        tool_calls: list[ToolCall] | None = None
        if msg.tool_calls:
            try:
                tool_calls = []
                for tc in msg.tool_calls:
                    if not isinstance(tc, ChatCompletionMessageToolCall):
                        _logger.warning("Unexpected tool call type %r — skipping", type(tc))
                        continue
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=json.loads(tc.function.arguments),
                        )
                    )
            except json.JSONDecodeError as exc:
                raise CompletionError(f"Malformed tool call arguments: {exc}") from exc
            tool_calls = tool_calls if tool_calls else None

        usage_data = response.usage
        usage = LLMUsage(
            input_tokens=usage_data.prompt_tokens if usage_data else 0,
            output_tokens=usage_data.completion_tokens if usage_data else 0,
        )

        return LLMResponse(
            message=Message(
                role=Role.ASSISTANT,
                content=msg.content,
                tool_calls=tool_calls,
            ),
            finish_reason=finish_reason,
            usage=usage,
        )
