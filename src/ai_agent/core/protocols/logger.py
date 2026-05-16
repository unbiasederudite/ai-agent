"""AgentLogger Protocol."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentLogger(Protocol):
    """Structured logger interface used throughout ``core/``.

    Each method emits one log record at the named level.  All extra keyword
    arguments are forwarded as structured fields on the record.

    Implementations must never raise — logging failures must be silently
    swallowed to avoid masking the original error.
    """

    def debug(self, event: str, **kwargs: object) -> None:
        """Emit a DEBUG-level record.

        Args:
            event: Short human-readable description of what happened.
            **kwargs: Additional structured fields attached to the record.
        """
        ...

    def info(self, event: str, **kwargs: object) -> None:
        """Emit an INFO-level record.

        Args:
            event: Short human-readable description of what happened.
            **kwargs: Additional structured fields attached to the record.
        """
        ...

    def warning(self, event: str, **kwargs: object) -> None:
        """Emit a WARNING-level record.

        Args:
            event: Short human-readable description of what happened.
            **kwargs: Additional structured fields attached to the record.
        """
        ...

    def error(self, event: str, **kwargs: object) -> None:
        """Emit an ERROR-level record.

        Args:
            event: Short human-readable description of what happened.
            **kwargs: Additional structured fields attached to the record.
        """
        ...
