"""Small process-owned circuit breaker for provider boundaries.

The breaker is deliberately not serializable and must never be placed in a
QueryState/checkpoint.  A process restart starts closed; durable Run/Outbox
state remains the source of truth for work recovery.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


class CircuitOpenError(OSError):
    """Raised when a provider call is rejected while its circuit is open."""


@dataclass(slots=True)
class CircuitState:
    """Bounded closed/open/half-open state owned by one process instance."""

    failure_threshold: int = 3
    reset_timeout_seconds: float = 30.0
    monotonic: Callable[[], float] = time.monotonic
    _failure_count: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _probe_in_flight: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if self.reset_timeout_seconds <= 0:
            raise ValueError("reset_timeout_seconds must be positive")

    @property
    def is_open(self) -> bool:
        """Return whether normal calls are currently short-circuited."""
        return self._opened_at is not None

    @property
    def failure_count(self) -> int:
        """Expose bounded local state for metrics without exposing providers."""
        return self._failure_count

    def allow_call(self) -> bool:
        """Allow closed calls and one half-open probe after the cool-down."""
        if self._opened_at is None:
            return True
        if self.monotonic() - self._opened_at < self.reset_timeout_seconds:
            return False
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def record_failure(self) -> bool:
        """Record one provider failure and return whether the circuit opened."""
        self._probe_in_flight = False
        self._failure_count += 1
        if self._failure_count < self.failure_threshold:
            return False
        newly_opened = self._opened_at is None
        self._opened_at = self.monotonic()
        return newly_opened

    def record_success(self) -> None:
        """Close the circuit after a successful normal call or probe."""
        self._failure_count = 0
        self._opened_at = None
        self._probe_in_flight = False


__all__ = ["CircuitOpenError", "CircuitState"]

