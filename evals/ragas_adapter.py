"""Explicit Ragas integration for offline batches with network-backed judges.

Ragas is intentionally an adapter rather than a dependency of the runner.  A
missing package or an unconfigured evaluator is represented explicitly as an
``unavailable`` result; no placeholder score is ever emitted.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
from typing import Any, Protocol, cast


class RagasUnavailable(RuntimeError):
    """Raised by callers that require Ragas instead of accepting unavailable."""


@dataclass(frozen=True, slots=True)
class RagasEvaluation:
    """JSON-safe outcome returned by the optional adapter."""

    status: str
    metrics: dict[str, float]
    reason: str | None = None
    metadata: dict[str, str | int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"status": self.status, "metrics": dict(self.metrics)}
        if self.reason is not None:
            result["reason"] = self.reason
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


class RagasBackend(Protocol):
    """Configured backend; real business judges call their model providers."""

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[str],
        reference_answer: str,
        answerable: bool = True,
    ) -> Mapping[str, object] | RagasEvaluation: ...


class RagasAdapter:
    """Call an explicitly configured backend or report it unavailable.

    ``backend`` is deliberately explicit.  Merely importing the optional
    package is not sufficient to configure a judge model and must not cause a
    network request.  A backend can be synchronous or asynchronous.
    """

    def __init__(self, backend: RagasBackend | Any | None = None) -> None:
        self._backend = backend

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def supports_refusal(self) -> bool:
        return getattr(self._backend, "supports_refusal", False) is True

    @property
    def fingerprint(self) -> str:
        return getattr(self._backend, "fingerprint", "unconfigured-ragas")

    async def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[str],
        reference_answer: str,
        answerable: bool = True,
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
            question=question,
            answer=answer,
            contexts=tuple(contexts),
            reference_answer=reference_answer,
            answerable=answerable,
        )
        if inspect.isawaitable(value):
            value = await value
        return normalize_ragas_result(value)


def normalize_ragas_result(value: object) -> RagasEvaluation:
    """Normalize and validate every adapter outcome at one shared boundary."""

    if isinstance(value, RagasEvaluation):
        return _validate_result(value)
    if not isinstance(value, Mapping):
        raise TypeError("offline Ragas backend must return a mapping")
    raw_metrics = value.get("metrics", value)
    if "metrics" in value:
        unknown = set(value) - {"status", "metrics", "reason", "metadata"}
        if unknown:
            raise ValueError(f"unknown Ragas result fields: {sorted(unknown)!r}")
    status = value.get("status", "available" if raw_metrics else "unavailable")
    if not isinstance(status, str) or status not in {"available", "unavailable", "failed"}:
        raise ValueError("Ragas status must be available, unavailable, or failed")
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("Ragas metrics must be a mapping")
    metrics: dict[str, float] = {}
    for name, metric in raw_metrics.items():
        if name in {"status", "reason", "metrics", "metadata"}:
            continue
        if not isinstance(name, str) or _METRIC_NAME.fullmatch(name.strip()) is None:
            raise ValueError("Ragas metric names must be safe non-empty strings")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError(f"Ragas metric {name!r} must be numeric")
        numeric = float(metric)
        if not math.isfinite(numeric):
            raise ValueError(f"Ragas metric {name!r} must be finite")
        metrics[name] = numeric
    reason = value.get("reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip() or len(reason) > 1_000):
        raise ValueError("Ragas reason must be a string")
    return _validate_result(RagasEvaluation(status=status, metrics=metrics, reason=reason,
                                           metadata=dict(value.get("metadata", {}))))


def _validate_result(value: RagasEvaluation) -> RagasEvaluation:
    if value.status not in {"available", "unavailable", "failed"}:
        raise ValueError("Ragas status must be available, unavailable, or failed")
    if value.status != "available" and value.metrics:
        raise ValueError("failed/unavailable Ragas results must not contain fake scores")
    if value.status == "available" and not value.metrics:
        raise ValueError("available Ragas results must contain metrics")
    for name, metric in value.metrics.items():
        if not isinstance(name, str) or _METRIC_NAME.fullmatch(name.strip()) is None:
            raise ValueError("Ragas metrics must have safe names")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            raise ValueError("Ragas metrics must be finite and named")
    if value.reason is not None and (
        not isinstance(value.reason, str) or not value.reason.strip() or len(value.reason) > 1_000
    ):
        raise ValueError("Ragas reason must be a bounded string")
    allowed_metadata = {"judge_model", "embedding_model", "embedding_dimensions", "ragas_version", "metrics_version"}
    if set(value.metadata) - allowed_metadata or any(
        type(v) not in {str, int} or (isinstance(v, str) and (not v.strip() or len(v) > 200))
        for v in value.metadata.values()
    ):
        raise ValueError("Ragas metadata must contain only bounded public version fields")
    return value


_METRIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")


__all__ = [
    "RagasBackend",
    "RagasAdapter",
    "RagasEvaluation",
    "RagasUnavailable",
    "normalize_ragas_result",
]
