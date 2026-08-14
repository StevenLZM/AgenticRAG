"""Projection of the durable Agent Event log into bounded online metrics."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.domain.models import UserScope
from agentic_rag.observability.logging import sanitize_attributes
from agentic_rag.persistence.artifacts import ArtifactStore
from agentic_rag.persistence.repositories import AgentEvent, EventRepository


class MetricsWindow(BaseModel):
    """A cursor-safe, snapshot-specific projection for one scoped run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    runtime_config_snapshot_id: str
    after_id: int
    through_id: int
    events_processed: int
    metrics: dict[str, float | int] = Field(default_factory=dict)


class MetricsProjector:
    """Read EventRepository rows and their safe artifacts without changing runtime."""

    def __init__(
        self,
        repository: EventRepository,
        artifacts: ArtifactStore | None,
        *,
        runtime_config_snapshot_id: str,
    ) -> None:
        if not runtime_config_snapshot_id.strip():
            raise ValueError("runtime_config_snapshot_id must not be blank")
        self._repository = repository
        self._artifacts = artifacts
        self._snapshot_id = runtime_config_snapshot_id

    async def project_window(
        self,
        *,
        run_id: str,
        scope: UserScope,
        after_id: int = 0,
        limit: int = 1_000,
    ) -> MetricsWindow:
        """Project one event cursor window for a Run owned by ``scope``.

        Event rows from another configuration snapshot advance the cursor but do
        not contribute data, preventing mixed-baseline totals during rollouts.
        """
        if after_id < 0 or limit <= 0:
            raise ValueError("after_id must be non-negative and limit positive")
        events = await self._repository.list_after(run_id, scope, after_id, limit)
        metrics = _empty_metrics()
        through_id = after_id
        run_started_at: datetime | None = None
        node_started_at: dict[str, datetime] = {}
        terminal_outcomes = 0
        outcome_counts = {"refusal": 0, "clarify": 0, "loop_limit": 0}
        citation_cited = 0.0
        citation_expected = 0.0
        processed = 0

        for event in events:
            through_id = max(through_id, event.id or after_id)
            if event.runtime_config_snapshot_id != self._snapshot_id:
                continue
            processed += 1
            event_type = event.event_type.upper()
            attributes = self._attributes_for(event)
            _add_number(metrics, "queue_wait_seconds", attributes.get("queue_wait_seconds"))
            _add_number(metrics, "run_latency_seconds", attributes.get("run_latency_seconds"))
            _add_number(metrics, "node_latency_seconds", attributes.get("node_latency_seconds"))
            _add_number(metrics, "retrieval_rounds", attributes.get("retrieval_rounds"))
            _add_number(metrics, "retrieval_candidates", attributes.get("candidate_count"))
            _add_number(metrics, "input_tokens", attributes.get("input_tokens"))
            _add_number(metrics, "output_tokens", attributes.get("output_tokens"))
            _add_number(metrics, "estimated_cost", attributes.get("estimated_cost"))
            _record_saturation(metrics, attributes)

            if event_type == "RUN_STARTED":
                run_started_at = event.created_at
            elif event_type == "GRAPH_NODE_STARTED" and event.node_name and event.created_at:
                node_started_at[event.node_name] = event.created_at
            elif event_type == "GRAPH_NODE_COMPLETED" and event.node_name and event.created_at:
                started = node_started_at.pop(event.node_name, None)
                if started is not None:
                    _add_number(metrics, "node_latency_seconds", (event.created_at - started).total_seconds())
            if event_type in {"RUN_COMPLETED", "RUN_FAILED", "RUN_CANCELLED"}:
                terminal_outcomes += 1
                if run_started_at is not None and event.created_at is not None:
                    _add_number(metrics, "run_latency_seconds", (event.created_at - run_started_at).total_seconds())

            termination = str(attributes.get("termination_reason", "")).casefold()
            is_refusal = "REFUS" in event_type or termination in {"refuse", "cannot_answer", "audit_failed"}
            is_clarify = "CLARIF" in event_type or termination == "clarify"
            is_loop_limit = "LOOP_LIMIT" in event_type or termination == "research_round_limit"
            if is_refusal:
                outcome_counts["refusal"] += 1
            if is_clarify:
                outcome_counts["clarify"] += 1
            if is_loop_limit:
                outcome_counts["loop_limit"] += 1
            if "REPAIR" in event_type:
                _add_number(metrics, "repair_count", 1)
            if "DEGRADED" in event_type or attributes.get("degraded_components"):
                _add_number(metrics, "degraded_component_count", 1)
            if "OUTBOX" in event_type and "REDISPATCH" in event_type:
                _add_number(metrics, "outbox_redispatch_count", 1)
            if "LEASE" in event_type and "RECLAIM" in event_type:
                _add_number(metrics, "lease_reclaim_count", 1)
            if "RECONCIL" in event_type and ("ANOMAL" in event_type or "ERROR" in event_type):
                _add_number(metrics, "reconciler_anomaly_count", 1)
            if event_type == "USER_FEEDBACK":
                _add_number(metrics, "feedback_count", 1)
            if "LLM" in event_type:
                _add_number(metrics, "model_call_count", 1)
            if "CITATION" in event_type:
                cited = _number(attributes.get("cited_claim_count"))
                expected = _number(attributes.get("claim_count"))
                if cited is not None:
                    citation_cited += cited
                if expected is not None:
                    citation_expected += expected

        metrics["citation_coverage"] = (
            citation_cited / citation_expected if citation_expected else 0.0
        )
        rate_denominator = terminal_outcomes or 1
        metrics["refusal_rate"] = outcome_counts["refusal"] / rate_denominator
        metrics["clarify_rate"] = outcome_counts["clarify"] / rate_denominator
        metrics["loop_limit_rate"] = outcome_counts["loop_limit"] / rate_denominator
        metrics["repair_rate"] = metrics["repair_count"] / rate_denominator
        metrics["degraded_component_rate"] = metrics["degraded_component_count"] / rate_denominator
        return MetricsWindow(
            run_id=run_id,
            runtime_config_snapshot_id=self._snapshot_id,
            after_id=after_id,
            through_id=through_id,
            events_processed=processed,
            metrics=metrics,
        )

    def _attributes_for(self, event: AgentEvent) -> dict[str, Any]:
        if event.payload_ref is None or self._artifacts is None:
            return {}
        try:
            ref = self._artifacts.describe(event.payload_ref)
            payload = self._artifacts.read_json(ref)
        except (OSError, ValueError, TypeError):
            return {}
        if not isinstance(payload, Mapping):
            return {}
        attributes = payload.get("attributes")
        return sanitize_attributes(attributes) if isinstance(attributes, Mapping) else {}


def _empty_metrics() -> dict[str, float | int]:
    return {
        "queue_wait_seconds": 0.0,
        "run_latency_seconds": 0.0,
        "node_latency_seconds": 0.0,
        "concurrent_saturation": 0.0,
        "retrieval_rounds": 0,
        "retrieval_candidates": 0,
        "refusal_rate": 0.0,
        "clarify_rate": 0.0,
        "repair_count": 0,
        "repair_rate": 0.0,
        "loop_limit_rate": 0.0,
        "degraded_component_count": 0,
        "degraded_component_rate": 0.0,
        "citation_coverage": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "model_call_count": 0,
        "estimated_cost": 0.0,
        "outbox_redispatch_count": 0,
        "lease_reclaim_count": 0,
        "reconciler_anomaly_count": 0,
        "feedback_count": 0,
    }


def _add_number(metrics: dict[str, float | int], key: str, value: object) -> None:
    number = _number(value)
    if number is None:
        return
    metrics[key] += number


def _number(value: object) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if value >= 0 else None


def _record_saturation(metrics: dict[str, float | int], attributes: Mapping[str, Any]) -> None:
    direct = _number(attributes.get("concurrent_saturation"))
    if direct is None:
        active = _number(attributes.get("concurrency_active"))
        limit = _number(attributes.get("concurrency_limit"))
        direct = active / limit if active is not None and limit is not None and limit > 0 else None
    if direct is not None:
        metrics["concurrent_saturation"] = max(metrics["concurrent_saturation"], direct)
