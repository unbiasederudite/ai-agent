"""Unit tests for ContextBudget."""

import pytest
from pydantic import ValidationError

from ai_agent.core.models.budget import ContextBudget
from ai_agent.core.models.llm import LLMUsage


def _usage(inp: int = 0, out: int = 0) -> LLMUsage:
    return LLMUsage(input_tokens=inp, output_tokens=out)


def _make(**overrides: object) -> ContextBudget:
    defaults: dict[str, object] = {"context_window": 1000}
    return ContextBudget(**{**defaults, **overrides})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


class TestContextBudgetConstruction:
    def test_constructs_with_context_window(self) -> None:
        b = _make()
        assert b.context_window == 1000

    def test_compaction_threshold_defaults_to_0_75(self) -> None:
        assert _make().compaction_threshold == 0.75

    def test_context_usage_defaults_to_zero(self) -> None:
        b = _make()
        assert b.context_usage.input_tokens == 0
        assert b.context_usage.output_tokens == 0

    def test_is_frozen(self) -> None:
        b = _make()
        with pytest.raises(Exception):
            b.context_window = 2000  # type: ignore[misc]

    def test_custom_threshold_stored(self) -> None:
        b = _make(compaction_threshold=0.5)
        assert b.compaction_threshold == 0.5

    def test_custom_context_usage_stored(self) -> None:
        b = _make(context_usage=_usage(inp=100, out=50))
        assert b.context_usage.input_tokens == 100
        assert b.context_usage.output_tokens == 50


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestContextBudgetValidation:
    def test_threshold_below_minimum_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make(compaction_threshold=0.09)

    def test_threshold_above_maximum_raises(self) -> None:
        with pytest.raises(ValidationError):
            _make(compaction_threshold=1.01)

    def test_threshold_at_minimum_is_valid(self) -> None:
        b = _make(compaction_threshold=0.1)
        assert b.compaction_threshold == 0.1

    def test_threshold_at_maximum_is_valid(self) -> None:
        b = _make(compaction_threshold=1.0)
        assert b.compaction_threshold == 1.0


# ---------------------------------------------------------------------------
# token_limit
# ---------------------------------------------------------------------------


class TestTokenLimit:
    def test_token_limit_is_floor_of_window_times_threshold(self) -> None:
        b = _make(context_window=1000, compaction_threshold=0.75)
        assert b.token_limit == 750

    def test_token_limit_truncates_fractional_result(self) -> None:
        b = _make(context_window=1000, compaction_threshold=0.333)
        assert b.token_limit == 333

    def test_token_limit_at_full_threshold(self) -> None:
        b = _make(context_window=500, compaction_threshold=1.0)
        assert b.token_limit == 500


# ---------------------------------------------------------------------------
# should_compact
# ---------------------------------------------------------------------------


class TestShouldCompact:
    def test_false_when_input_tokens_zero(self) -> None:
        b = _make(context_usage=_usage(inp=0, out=900))
        assert b.should_compact is False

    def test_false_when_usage_below_limit(self) -> None:
        b = _make(
            context_window=1000, compaction_threshold=0.75, context_usage=_usage(inp=500, out=100)
        )
        assert b.should_compact is False

    def test_false_when_usage_exactly_at_limit(self) -> None:
        b = _make(
            context_window=1000, compaction_threshold=0.75, context_usage=_usage(inp=700, out=50)
        )
        assert b.should_compact is False

    def test_true_when_usage_exceeds_limit(self) -> None:
        b = _make(
            context_window=1000, compaction_threshold=0.75, context_usage=_usage(inp=700, out=51)
        )
        assert b.should_compact is True

    def test_false_on_fresh_budget(self) -> None:
        assert _make().should_compact is False


# ---------------------------------------------------------------------------
# utilization
# ---------------------------------------------------------------------------


class TestUtilization:
    def test_zero_when_no_usage(self) -> None:
        assert _make().utilization == 0.0

    def test_zero_when_context_window_is_zero(self) -> None:
        b = ContextBudget(context_window=0, context_usage=_usage(inp=100, out=0))
        assert b.utilization == 0.0

    def test_utilization_is_total_over_window(self) -> None:
        b = _make(context_window=1000, context_usage=_usage(inp=600, out=200))
        assert b.utilization == pytest.approx(0.8)

    def test_utilization_can_exceed_one(self) -> None:
        b = _make(context_window=100, context_usage=_usage(inp=90, out=20))
        assert b.utilization > 1.0


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_returns_new_instance(self) -> None:
        b = _make()
        b2 = b.update(_usage(inp=100, out=50))
        assert b2 is not b

    def test_context_usage_is_updated(self) -> None:
        b = _make().update(_usage(inp=200, out=80))
        assert b.context_usage.input_tokens == 200
        assert b.context_usage.output_tokens == 80

    def test_original_is_unchanged(self) -> None:
        b = _make()
        b.update(_usage(inp=500, out=100))
        assert b.context_usage.input_tokens == 0

    def test_other_fields_preserved(self) -> None:
        b = _make(context_window=5000, compaction_threshold=0.9)
        b2 = b.update(_usage(inp=10, out=5))
        assert b2.context_window == 5000
        assert b2.compaction_threshold == 0.9


# ---------------------------------------------------------------------------
# reset_usage
# ---------------------------------------------------------------------------


class TestResetUsage:
    def test_returns_new_instance(self) -> None:
        b = _make(context_usage=_usage(inp=100, out=50))
        assert b.reset_usage() is not b

    def test_context_usage_is_zeroed(self) -> None:
        b = _make(context_usage=_usage(inp=500, out=200)).reset_usage()
        assert b.context_usage.input_tokens == 0
        assert b.context_usage.output_tokens == 0

    def test_original_is_unchanged(self) -> None:
        b = _make(context_usage=_usage(inp=300, out=100))
        b.reset_usage()
        assert b.context_usage.input_tokens == 300

    def test_other_fields_preserved(self) -> None:
        b = _make(
            context_window=8000, compaction_threshold=0.6, context_usage=_usage(inp=100, out=0)
        )
        b2 = b.reset_usage()
        assert b2.context_window == 8000
        assert b2.compaction_threshold == 0.6

    def test_should_compact_false_after_reset(self) -> None:
        b = _make(context_window=100, compaction_threshold=0.5, context_usage=_usage(inp=80, out=0))
        assert b.reset_usage().should_compact is False


# ---------------------------------------------------------------------------
# recalibrate
# ---------------------------------------------------------------------------


class TestRecalibrate:
    def test_returns_new_instance(self) -> None:
        b = _make()
        assert b.recalibrate(2000) is not b

    def test_context_window_is_updated(self) -> None:
        b = _make(context_window=1000).recalibrate(4000)
        assert b.context_window == 4000

    def test_original_is_unchanged(self) -> None:
        b = _make(context_window=1000)
        b.recalibrate(4000)
        assert b.context_window == 1000

    def test_threshold_and_usage_preserved(self) -> None:
        b = _make(
            context_window=1000,
            compaction_threshold=0.6,
            context_usage=_usage(inp=50, out=10),
        )
        b2 = b.recalibrate(2000)
        assert b2.compaction_threshold == 0.6
        assert b2.context_usage.input_tokens == 50

    def test_token_limit_reflects_new_window(self) -> None:
        b = _make(context_window=1000, compaction_threshold=0.5).recalibrate(2000)
        assert b.token_limit == 1000
