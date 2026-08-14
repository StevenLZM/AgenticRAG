"""Optional, strictly offline Ragas integration.

Ragas is intentionally an adapter rather than a dependency of the runner.  A
missing package or an unconfigured evaluator is represented explicitly as an
``unavailable`` result; no placeholder score is ever emitted.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast


class RagasUnavailable(RuntimeError):
    """Raised by callers that require Ragas instead of accepting unavailable."""


@dataclass(frozen=True, slots=True)
class RagasEvaluation:
    """JSON-safe outcome returned by the optional adapter."""

    status: str
    metrics: dict[str, float]
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"status": self.status, "metrics": dict(self.metrics)}
        if self.reason is not None:
            result["reason"] = self.reason
        return result


class OfflineRagasBackend(Protocol):
    """Minimal dependency-injected backend; it must not perform network I/O."""

    def evaluate(
        self,
        *,
        answer: str,
        contexts: Sequence[str],
        reference_answer: str,
    ) -> Mapping[str, object] | RagasEvaluation: ...


class RagasAdapter:
    """Call an injected offline backend or report that Ragas is unavailable.

    ``backend`` is deliberately explicit.  Merely importing the optional
    package is not sufficient to configure a judge model and must not cause a
    network request.  A backend can be synchronous or asynchronous.
    """

    def __init__(self, backend: OfflineRagasBackend | Any | None = None) -> None:
        self._backend = backend

    @property
    def available(self) -> bool:
        return self._backend is not None

    async def evaluate(
        self,
        *,
        answer: str,
        contexts: Sequence[str],
        reference_answer: str,
    ) -> RagasEvaluation:
        backend = self._backend
        if backend is None:
            return RagasEvaluation(
                status="unavailable",
                metrics={},
                reason="ragas backend is not installed or configured",
            )
        evaluator = getattr(backend, "evaluate", backend)
        if not callable(evaluator):
            raise TypeError("offline Ragas backend must expose evaluate or be callable")
        invoke = cast(Callable[..., object], evaluator)
        value = invoke(
            answer=answer,
            contexts=tuple(contexts),
            reference_answer=reference_answer,
        )
        if inspect.isawaitable(value):
            value = await value
        return _normalize_result(value)


def _normalize_result(value: object) -> RagasEvaluation:
    if isinstance(value, RagasEvaluation):
        return _validate_result(value)
    if not isinstance(value, Mapping):
        raise TypeError("offline Ragas backend must return a mapping")
    status = value.get("status", "available")
    if not isinstance(status, str) or status not in {"available", "unavailable"}:
        raise ValueError("Ragas status must be available or unavailable")
    raw_metrics = value.get("metrics", value)
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("Ragas metrics must be a mapping")
    metrics: dict[str, float] = {}
    for name, metric in raw_metrics.items():
        if name in {"status", "reason", "metrics"}:
            continue
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Ragas metric names must be non-empty strings")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError(f"Ragas metric {name!r} must be numeric")
        numeric = float(metric)
        if numeric != numeric or numeric in {float("inf"), float("-inf")}:
            raise ValueError(f"Ragas metric {name!r} must be finite")
        metrics[name] = numeric
    reason = value.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("Ragas reason must be a string")
    return _validate_result(RagasEvaluation(status=status, metrics=metrics, reason=reason))


def _validate_result(value: RagasEvaluation) -> RagasEvaluation:
    if value.status not in {"available", "unavailable"}:
        raise ValueError("Ragas status must be available or unavailable")
    if value.status == "unavailable" and value.metrics:
        raise ValueError("unavailable Ragas results must not contain fake scores")
    if value.status == "available" and not value.metrics:
        raise ValueError("available Ragas results must contain metrics")
    for name, metric in value.metrics.items():
        if not name.strip() or metric != metric or metric in {float("inf"), float("-inf")}:
            raise ValueError("Ragas metrics must be finite and named")
    return value


__all__ = ["OfflineRagasBackend", "RagasAdapter", "RagasEvaluation", "RagasUnavailable"]
