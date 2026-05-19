"""LLM-based session history compaction."""

from __future__ import annotations

import logging

from ai_agent.core.exceptions import CompletionError
from ai_agent.core.models.llm import CompactionResult, LLMRequest, LLMSettings, LLMUsage
from ai_agent.core.models.message import Message, Role
from ai_agent.core.protocols.llm import ILLMProvider

_log = logging.getLogger(__name__)


_COMPACTION_PROMPT = (
    "Summarize the conversation above into a concise paragraph. "
    "Preserve all key decisions, conclusions, facts, and context required to continue the conversation."
)


class CompactionService:
    """Summarises old turns and returns a compacted message list."""

    def __init__(
        self,
        provider: ILLMProvider,
        model: str,
        settings: LLMSettings,
        compaction_prompt: str = _COMPACTION_PROMPT,
    ) -> None:
        self._provider = provider
        self._model = model
        self._settings = settings
        self._compaction_prompt = compaction_prompt

    def compact(self, messages: list[Message], keep_recent_turns: int = 3) -> CompactionResult:
        """Summarise old turns and return a compacted message list with token usage.

        Args:
            messages: Current message list, may include a leading SYSTEM summary.
            keep_recent_turns: Number of complete turns to preserve verbatim.

        Returns:
            CompactionResult with [SYSTEM summary, ...recent turns] and LLM usage.
            If nothing to compact, messages is returned unchanged and usage is None.

        Raises:
            CompletionError: LLM returned an empty summary.
        """
        existing_summary = next((m for m in messages if m.role == Role.SYSTEM), None)
        conversation = [m for m in messages if m.role != Role.SYSTEM]

        turns = self._split_into_turns(conversation)
        if len(turns) <= keep_recent_turns:
            return CompactionResult(messages=messages)

        to_summarize = turns[:-keep_recent_turns]
        to_keep = turns[-keep_recent_turns:]

        summary_input: list[Message] = []
        if existing_summary is not None:
            summary_input.append(existing_summary)
        summary_input.extend(m for turn in to_summarize for m in turn)

        summary_text, usage = self._summarize(summary_input)

        _log.info("compaction.complete", extra={"kept_turns": len(to_keep)})
        return CompactionResult(
            messages=[
                Message(role=Role.SYSTEM, content=summary_text),
                *(m for turn in to_keep for m in turn),
            ],
            usage=usage,
        )

    def _summarize(self, messages: list[Message]) -> tuple[str, LLMUsage]:
        request_messages = [
            *messages,
            Message(role=Role.USER, content=self._compaction_prompt),
        ]
        response = self._provider.complete(
            LLMRequest(model=self._model, settings=self._settings, messages=request_messages)
        )
        _log.info(
            "compaction.llm_call",
            extra={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
        if not response.message.content:
            raise CompletionError("Compaction produced an empty summary.")
        return response.message.content, response.usage

    @staticmethod
    def _split_into_turns(messages: list[Message]) -> list[list[Message]]:
        turns: list[list[Message]] = []
        current_turn: list[Message] = []
        for message in messages:
            if message.role == Role.USER and current_turn:
                turns.append(current_turn)
                current_turn = []
            current_turn.append(message)
        if current_turn:
            turns.append(current_turn)
        return turns
