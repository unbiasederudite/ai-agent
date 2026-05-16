"""Unit tests for the AgentLogger Protocol.

Tests use structural subtyping: a concrete stub that satisfies the Protocol
must be accepted by a function typed to receive ``AgentLogger``.
"""

from typing import Any

from ai_agent.core.protocols import AgentLogger


class _StubLogger:
    """Minimal concrete class that satisfies the AgentLogger Protocol."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def debug(self, event: str, **kwargs: object) -> None:
        self.records.append({"level": "debug", "event": event, **kwargs})

    def info(self, event: str, **kwargs: object) -> None:
        self.records.append({"level": "info", "event": event, **kwargs})

    def warning(self, event: str, **kwargs: object) -> None:
        self.records.append({"level": "warning", "event": event, **kwargs})

    def error(self, event: str, **kwargs: object) -> None:
        self.records.append({"level": "error", "event": event, **kwargs})


def _accepts_logger(logger: AgentLogger) -> str:
    """Helper typed to receive AgentLogger — used to check structural compat."""
    logger.info("test_event", key="value")
    return "ok"


class TestAgentLoggerProtocol:
    """Tests for AgentLogger Protocol structural subtyping."""

    def test_stub_satisfies_protocol(self) -> None:
        stub = _StubLogger()
        assert _accepts_logger(stub) == "ok"

    def test_debug_method_exists(self) -> None:
        stub = _StubLogger()
        stub.debug("startup", component="core")
        assert stub.records[0]["level"] == "debug"
        assert stub.records[0]["event"] == "startup"

    def test_info_method_exists(self) -> None:
        stub = _StubLogger()
        stub.info("turn_start", turn=1)
        assert stub.records[0]["level"] == "info"

    def test_warning_method_exists(self) -> None:
        stub = _StubLogger()
        stub.warning("eviction", key="old_key")
        assert stub.records[0]["level"] == "warning"

    def test_error_method_exists(self) -> None:
        stub = _StubLogger()
        stub.error("tool_failed", tool="calc")
        assert stub.records[0]["level"] == "error"

    def test_kwargs_forwarded(self) -> None:
        stub = _StubLogger()
        stub.info("run_complete", turns=5, status="complete")
        assert stub.records[0]["turns"] == 5
        assert stub.records[0]["status"] == "complete"
