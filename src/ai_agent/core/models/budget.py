"""Context window budget tracking."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ai_agent.core.models.llm import LLMUsage


class ContextBudget(BaseModel):
    """Tracks context window consumption and determines compaction need."""

    model_config = ConfigDict(frozen=True)

    context_window: int = Field(
        description="Total context window size of the active LLM.",
    )
    compaction_threshold: float = Field(
        default=0.75,
        ge=0.1,
        le=1.0,
        description="Compact when context reaches this fraction of the context window.",
    )
    context_usage: LLMUsage = Field(
        default_factory=lambda: LLMUsage(input_tokens=0, output_tokens=0),
        description="Token usage from the last run's final step.",
    )

    @property
    def token_limit(self) -> int:
        """Token count at which compaction should trigger."""
        return int(self.context_window * self.compaction_threshold)

    @property
    def should_compact(self) -> bool:
        """True when context usage exceeds the token limit."""
        if self.context_usage.input_tokens == 0:
            return False
        return self.context_usage.total_tokens > self.token_limit

    @property
    def utilization(self) -> float:
        """Context utilization as 0.0-1.0 fraction."""
        if self.context_window == 0:
            return 0.0
        return self.context_usage.total_tokens / self.context_window

    def update(self, context_usage: LLMUsage) -> ContextBudget:
        """Return updated budget after a run completes."""
        return self.model_copy(update={"context_usage": context_usage})

    def reset_usage(self) -> ContextBudget:
        """Reset usage after compaction — next run remeasures."""
        return self.model_copy(
            update={
                "context_usage": LLMUsage(input_tokens=0, output_tokens=0),
            }
        )

    def recalibrate(self, context_window: int) -> ContextBudget:
        """Return new budget recalibrated for a different LLM."""
        return self.model_copy(update={"context_window": context_window})
