"""Lease-fenced Redis Stream worker for durable checkpointed query runs."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import Literal, Protocol, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import ValidationError

from agentic_rag.query.answer_sources import build_answer_sources, InvalidAnswerSources
from agentic_rag.query.evidence_builder import PackedEvidence

from agentic_rag.domain.models import RunStatus, UserScope
from agentic_rag.observability.logging import (
    AgentEventEmitter,
    emit_degradation,
    event_emission_scope,
    stable_event_key,
)
from agentic_rag.observability.tracing import TraceRecorder
from agentic_rag.persistence.outbox import OutboxDispatcher
from agentic_rag.persistence.redis_queue import StreamBroker, StreamMessage
from agentic_rag.persistence.repositories import LeaseLost, QueryRun, RunRepository
from agentic_rag.query.public_answer import project_public_answer
from agentic_rag.query.state import new_query_state
from agentic_rag.runtime.concurrency import ConcurrencyManager
from agentic_rag.runtime.models import RuntimeConfigSnapshot


QUERY_STREAM = "agenticrag:jobs:query"
QUERY_GROUP = "agenticrag-query-workers"
QUERY_DEAD_STREAM = "agenticrag:jobs:query:dead"
logger = logging.getLogger(__name__)
TERMINAL_RUN_STATUSES = {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED}
BUSINESS_TERMINAL_REASONS = frozenset(
    {
        "completed",
        "clarify",
        "refuse",
        "cannot_answer",
        "research_action_invalid",
        "audit_failed",
        "research_round_limit",
    }
)


class QueryGraph(Protocol):
    async def ainvoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]: ...


QueryGraphFactory = Callable[..., Awaitable[QueryGraph] | QueryGraph]


class RunCancelled(RuntimeError):
    """The durable cancellation flag was observed by the worker heartbeat."""


def build_graph_factory(
    dependencies: object, checkpointer: BaseCheckpointSaver[str]
) -> QueryGraphFactory:
    """Bind process-owned graph services to one durable checkpoint backend.

    The message only selects its server-derived checkpoint thread id; it never
    supplies services or a checkpointer as part of Redis-delivered state.
    """
    from agentic_rag.query.graph import QueryGraphDependencies, build_query_graph

    if not isinstance(dependencies, QueryGraphDependencies):
        raise TypeError("dependencies must be QueryGraphDependencies")

    def factory(*, checkpoint_thread_id: str) -> QueryGraph:
        if not checkpoint_thread_id.startswith("query:"):
            raise ValueError("query checkpoint thread_id must use query namespace")
        return cast(QueryGraph, build_query_graph(dependencies, checkpointer))

    return factory


class QueryWorker:
    """Claim once, resume the fixed graph checkpoint, then ACK terminal runs only."""

    def __init__(
        self,
        *,
        runs: RunRepository,
        broker: StreamBroker,
        graph_factory: QueryGraphFactory,
        worker_id: str,
        concurrency: ConcurrencyManager | None = None,
        lease_seconds: int = 30,
        heartbeat_interval_seconds: float = 10,
        run_timeout_seconds: float = 300,
        reclaim_idle_ms: int = 30_000,
        block_ms: int = 1_000,
        max_attempts: int = 3,
        trace_recorder: TraceRecorder | None = None,
        event_emitter: AgentEventEmitter | None = None,
        outbox_dispatcher: OutboxDispatcher | None = None,
        outbox_interval_seconds: float = 1.0,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be blank")
        if lease_seconds <= 0 or heartbeat_interval_seconds <= 0:
            raise ValueError("lease intervals must be positive")
        if heartbeat_interval_seconds >= lease_seconds:
            raise ValueError("heartbeat interval must be shorter than lease")
        if run_timeout_seconds <= 0 or reclaim_idle_ms < 0 or block_ms < 0:
            raise ValueError("worker timeouts must be bounded")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if outbox_interval_seconds <= 0:
            raise ValueError("outbox interval must be positive")
        self._runs = runs
        self._broker = broker
        self._graph_factory = graph_factory
        self._worker_id = worker_id
        self._concurrency = concurrency or ConcurrencyManager()
        self._lease_seconds = lease_seconds
        self._heartbeat_interval = heartbeat_interval_seconds
        self._run_timeout = run_timeout_seconds
        self._reclaim_idle_ms = reclaim_idle_ms
        self._block_ms = block_ms
        self._max_attempts = max_attempts
        self._trace_recorder = trace_recorder
        self._event_emitter = event_emitter
        self._outbox_dispatcher = outbox_dispatcher
        self._outbox_interval = outbox_interval_seconds

    async def run_one(self) -> bool:
        """Drain one Redis delivery batch, preferring reclaimed messages first."""
        reclaimed = await self._broker.reclaim(
            QUERY_STREAM, QUERY_GROUP, self._worker_id, self._reclaim_idle_ms
        )
        fresh = await self._broker.consume(
            QUERY_STREAM, QUERY_GROUP, self._worker_id, self._block_ms
        )
        messages = (*reclaimed, *fresh)
        if not messages:
            return False
        for message in messages:
            await self.process_message(message)
        return True

    async def process_message(self, message: StreamMessage) -> None:
        claim = await self._runs.claim(message.aggregate_id, self._worker_id, self._lease_seconds)
        if claim is None:
            existing = await self._runs.get_for_delivery(message.aggregate_id)
            if existing is not None and existing.status in TERMINAL_RUN_STATUSES:
                await self._broker.ack(QUERY_STREAM, QUERY_GROUP, message.id)
            return
        snapshot = RuntimeConfigSnapshot.model_validate(claim.runtime_config_snapshot)
        run_started = time.perf_counter()
        await self._emit_run_lifecycle(
            claim,
            "RUN_STARTED",
            started_monotonic=run_started,
        )
        queue_wait = max(0.0, (datetime.now(UTC) - message.enqueued_at).total_seconds())
        if (
            self._event_emitter is not None
            and self._event_emitter.runtime_config_snapshot_id == snapshot.snapshot_id
        ):
            try:
                await self._event_emitter.emit(
                    run_id=claim.id,
                    user_id=claim.user_id,
                    event_type="QUEUE_WAITED",
                    summary="completed",
                    event_key=stable_event_key(
                        claim.id, message.id, "QUEUE_WAITED"
                    ),
                    attributes={"queue_wait_seconds": queue_wait},
                )
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                pass
        recorder = (
            self._trace_recorder
            if self._trace_recorder is not None
            and self._trace_recorder.runtime_config_snapshot_id == snapshot.snapshot_id
            else None
        )
        emitter = (
            self._event_emitter
            if self._event_emitter is not None
            and self._event_emitter.runtime_config_snapshot_id == snapshot.snapshot_id
            else None
        )
        if emitter is not None:
            async with event_emission_scope(emitter, claim.id, "queue", user_id=claim.user_id):
                if recorder is not None:
                    async with recorder.span(
                        "queue", run_id=claim.id, attributes={"queue_wait_seconds": queue_wait}
                    ):
                        await self._process_claimed_message(
                            message, claim, started_monotonic=run_started
                        )
                else:
                    await self._process_claimed_message(
                        message, claim, started_monotonic=run_started
                    )
            return
        if recorder is not None:
            async with recorder.span(
                "queue", run_id=claim.id, attributes={"queue_wait_seconds": queue_wait}
            ):
                await self._process_claimed_message(
                    message, claim, started_monotonic=run_started
                )
            return
        await self._process_claimed_message(message, claim, started_monotonic=run_started)

    async def _process_claimed_message(
        self,
        message: StreamMessage,
        claim: QueryRun,
        *,
        started_monotonic: float,
    ) -> None:
        if claim.status is RunStatus.CANCEL_REQUESTED:
            await self._finish_and_ack(
                message,
                claim,
                RunStatus.CANCELLED,
                None,
                started_monotonic=started_monotonic,
                termination_reason="cancelled",
            )
            return
        if not claim.question.strip():
            await self._emit_degradation(
                claim,
                "invalid_input",
                attempt=claim.claim_generation,
                retryable=False,
                outcome="refused",
                event_type="QUERY_REFUSED",
            )
            await self._finish_and_ack(
                message,
                claim,
                RunStatus.FAILED,
                "missing_query",
                started_monotonic=started_monotonic,
                termination_reason="failed",
            )
            return
        try:
            async with self._concurrency.run_slot():
                async with asyncio.timeout(
                    min(self._run_timeout, float(RuntimeConfigSnapshot.model_validate(claim.runtime_config_snapshot).query_run_timeout_seconds))
                ):
                    result = await self._invoke_with_heartbeat(claim)
        except RunCancelled:
            await self._emit_degradation(
                claim, "cancelled", attempt=claim.claim_generation, retryable=False,
                outcome="refused", event_type="QUERY_CANCELLED"
            )
            await self._finish_and_ack(
                message,
                claim,
                RunStatus.CANCELLED,
                None,
                started_monotonic=started_monotonic,
                termination_reason="cancelled",
            )
            return
        except asyncio.TimeoutError:
            await self._emit_degradation(
                claim, "worker_timeout", attempt=claim.claim_generation, retryable=True,
                outcome="degraded", event_type="QUERY_TIMEOUT"
            )
            await self._finish_and_ack(
                message,
                claim,
                RunStatus.FAILED,
                "run_timeout",
                started_monotonic=started_monotonic,
                termination_reason="failed",
            )
            return
        except LeaseLost:
            await self._emit_degradation(
                claim, "lease_lost", attempt=claim.claim_generation, retryable=True,
                outcome="degraded", event_type="LEASE_LOST"
            )
            return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._handle_execution_failure(
                message,
                claim,
                type(error).__name__,
                started_monotonic=started_monotonic,
            )
            return

        if not isinstance(result, dict):
            await self._handle_execution_failure(
                message,
                claim,
                "invalid_graph_result",
                started_monotonic=started_monotonic,
            )
            return
        termination = result.get("termination_reason")
        if not isinstance(termination, str):
            await self._handle_execution_failure(
                message,
                claim,
                "invalid_termination",
                started_monotonic=started_monotonic,
            )
            return
        if termination == "cancelled":
            terminal = RunStatus.CANCELLED
        elif termination in BUSINESS_TERMINAL_REASONS:
            terminal = RunStatus.COMPLETED
        else:
            await self._handle_execution_failure(
                message,
                claim,
                "invalid_termination",
                started_monotonic=started_monotonic,
            )
            return
        answer = result.get("answer")
        if answer is None and terminal is RunStatus.COMPLETED:
            answer = {"status": termination}
        if not isinstance(answer, dict) and answer is not None:
            await self._handle_execution_failure(
                message,
                claim,
                "invalid_graph_result",
                started_monotonic=started_monotonic,
            )
            return
        if isinstance(answer, dict):
            if termination == "completed":
                answer.pop("status", None)
            else:
                answer["status"] = termination
            answer = _public_answer_projection(
                answer,
                result,
                runtime_config_snapshot_id=claim.runtime_config_snapshot_id,
                require_audited=termination == "completed",
            )
            if answer is None:
                await self._handle_execution_failure(
                    message,
                    claim,
                    "invalid_graph_result",
                    started_monotonic=started_monotonic,
                )
                return
        sources = None
        if terminal is RunStatus.COMPLETED and answer is not None:
            try:
                public = project_public_answer(answer)
                packed = PackedEvidence.model_validate(result.get("packed_context"))
                projection = build_answer_sources(public, packed, run_id=claim.id,
                    snapshot_id=claim.runtime_config_snapshot_id) if public else None
                sources = projection.model_dump(mode="json") if projection else None
            except (ValidationError, InvalidAnswerSources, TypeError):
                # Optional display projection cannot invalidate an audited answer.
                sources = None
        await self._finish_and_ack(
            message,
            claim,
            terminal,
            None,
            started_monotonic=started_monotonic,
            termination_reason=termination,
            answer=cast(dict[str, object] | None, answer),
            answer_sources=sources,
        )
    async def run_forever(self, *, stop_event: asyncio.Event | None = None) -> None:
        stop = stop_event or asyncio.Event()
        current: asyncio.Task[bool] | None = None
        dispatcher_task: asyncio.Task[None] | None = None
        failures = 0
        try:
            if self._outbox_dispatcher is not None:
                dispatcher_task = asyncio.create_task(
                    self._dispatch_outbox_forever(stop)
                )
            while not stop.is_set():
                current = asyncio.create_task(self.run_one())
                try:
                    await asyncio.shield(current)
                    failures = 0
                except asyncio.CancelledError:
                    stop.set()
                    with suppress(asyncio.CancelledError):
                        await current
                    raise
                except Exception:
                    failures += 1
                    await asyncio.sleep(min(0.25 * (2 ** min(failures - 1, 4)), 5.0))
                finally:
                    current = None
        finally:
            if current is not None:
                with suppress(asyncio.CancelledError):
                    await current
            if dispatcher_task is not None:
                dispatcher_task.cancel()
                with suppress(asyncio.CancelledError):
                    await dispatcher_task

    async def _dispatch_outbox_forever(self, stop: asyncio.Event) -> None:
        assert self._outbox_dispatcher is not None
        while not stop.is_set():
            try:
                await self._outbox_dispatcher.dispatch_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("query_outbox_dispatch_failed")
                await emit_degradation(
                    component="outbox",
                    reason="outbox_retry",
                    run_id=None,
                    snapshot_id="",
                    attempt=1,
                    retryable=True,
                    outcome="degraded",
                    event_type="OUTBOX_RETRY",
                )
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._outbox_interval)
            except asyncio.TimeoutError:
                continue

    async def _invoke_with_heartbeat(self, claim: QueryRun) -> dict[str, object]:
        graph = self._graph_factory(checkpoint_thread_id=claim.checkpoint_thread_id)
        if hasattr(graph, "__await__"):
            graph = await graph  # type: ignore[assignment,misc]
        state = new_query_state(
            run_id=claim.id, question=claim.question, scope=UserScope(user_id=claim.user_id),
            snapshot=RuntimeConfigSnapshot.model_validate(claim.runtime_config_snapshot),
            # Memory extraction consumes public messages, not request.question.
            # Keep provenance stable when this run is invoked again.
            messages=[{
                "id": f"query:{claim.id}:user",
                "role": "user",
                "content": claim.question,
            }],
        )
        config: dict[str, object] = {"configurable": {"thread_id": claim.checkpoint_thread_id}}
        state["request"]["thread_id"] = claim.thread_id
        task = asyncio.create_task(graph.ainvoke(dict(state), config))  # type: ignore[union-attr]
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self._heartbeat_interval)
                if done:
                    return await task
                current = await self._runs.get(
                    claim.id, UserScope(user_id=claim.user_id)
                )
                if current is None:
                    raise LeaseLost(claim.id)
                if current.status in {RunStatus.CANCEL_REQUESTED, RunStatus.CANCELLED}:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    raise RunCancelled(claim.id)
                await self._runs.heartbeat(
                    claim.id, self._worker_id, self._lease_seconds,
                    claim_generation=claim.claim_generation,
                )
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def _finish_and_ack(
        self,
        message: StreamMessage,
        claim: QueryRun,
        status: RunStatus,
        error_code: str | None,
        *,
        started_monotonic: float,
        termination_reason: str,
        answer: dict[str, object] | None = None,
        answer_sources: dict[str, object] | None = None,
    ) -> None:
        try:
            await self._runs.finish(
                claim.id, status, None, error_code,
                owner=self._worker_id, claim_generation=claim.claim_generation,
                answer=answer, answer_sources=answer_sources,
            )
        except LeaseLost:
            # The lease fence is authoritative.  Leave the Redis delivery
            # pending for reclaim and do not append or acknowledge a terminal
            # event owned by a different worker.
            await self._emit_degradation(
                claim,
                "lease_lost",
                attempt=claim.claim_generation,
                retryable=True,
                outcome="degraded",
                event_type="LEASE_LOST",
            )
            return
        await self._emit_run_lifecycle(
            claim,
            _run_event_type(status),
            started_monotonic=started_monotonic,
            termination_reason=termination_reason,
        )
        await self._broker.ack(QUERY_STREAM, QUERY_GROUP, message.id)

    async def _dead_letter_and_fail(
        self,
        message: StreamMessage,
        claim: QueryRun,
        reason: str,
        *,
        started_monotonic: float,
    ) -> None:
        await self._emit_degradation(
            claim,
            "worker_dlq",
            attempt=claim.claim_generation,
            retryable=False,
            outcome="dlq",
            event_type="WORKER_DLQ",
        )
        await self._broker.dead_letter(
            QUERY_DEAD_STREAM, message, reason,
            dedupe_key=f"query-dead:{claim.id}:{claim.claim_generation}",
        )
        await self._finish_and_ack(
            message,
            claim,
            RunStatus.FAILED,
            reason,
            started_monotonic=started_monotonic,
            termination_reason="failed",
        )

    async def _handle_execution_failure(
        self,
        message: StreamMessage,
        claim: QueryRun,
        reason: str,
        *,
        started_monotonic: float,
    ) -> None:
        if claim.claim_generation >= self._max_attempts:
            await self._dead_letter_and_fail(
                message,
                claim,
                reason,
                started_monotonic=started_monotonic,
            )
        else:
            await self._emit_degradation(
                claim,
                "provider_outage",
                attempt=claim.claim_generation,
                retryable=True,
                outcome="degraded",
                event_type="QUERY_RETRY",
            )

    async def _emit_degradation(
        self,
        claim: QueryRun,
        reason: str,
        *,
        attempt: int,
        retryable: bool,
        outcome: Literal["degraded", "refused", "dlq"],
        event_type: str,
    ) -> None:
        await emit_degradation(
            component="query_worker",
            reason=reason,
            run_id=claim.id,
            snapshot_id=RuntimeConfigSnapshot.model_validate(
                claim.runtime_config_snapshot
            ).snapshot_id,
            attempt=attempt,
            retryable=retryable,
            outcome=outcome,
            event_type=event_type,
        )

    async def _emit_run_lifecycle(
        self,
        claim: QueryRun,
        event_type: str,
        *,
        started_monotonic: float,
        termination_reason: str | None = None,
    ) -> None:
        """Emit authoritative run lifecycle metadata after the lease boundary.

        ``RUN_STARTED`` is written immediately after a successful claim.  A
        terminal event is written only after ``RunRepository.finish`` returns,
        so a stale owner that loses its lease cannot publish an authoritative
        outcome.  The event key is stable across Redis redelivery and the
        measured duration is carried as a trusted, server-observed attribute.
        """
        emitter = self._event_emitter
        if emitter is None:
            return
        try:
            snapshot_id = RuntimeConfigSnapshot.model_validate(
                claim.runtime_config_snapshot
            ).snapshot_id
        except (TypeError, ValueError):
            return
        if emitter.runtime_config_snapshot_id != snapshot_id:
            return
        attributes: dict[str, object] = {}
        if event_type != "RUN_STARTED":
            attributes["run_latency_seconds"] = max(
                0.0, time.perf_counter() - started_monotonic
            )
            if termination_reason in {
                "completed",
                "clarify",
                "refuse",
                "cannot_answer",
                "research_action_invalid",
                "audit_failed",
                "research_round_limit",
                "cancelled",
                "failed",
            }:
                attributes["termination_reason"] = termination_reason
        summary = {
            "RUN_STARTED": "started",
            "RUN_COMPLETED": "completed",
            "RUN_FAILED": "failed",
            "RUN_CANCELLED": "cancelled",
        }.get(event_type, "telemetry event")
        try:
            await emitter.emit(
                run_id=claim.id,
                user_id=claim.user_id,
                event_type=event_type,
                summary=summary,
                # A Run has one durable lifecycle even when Redis reclaims a
                # delivery and increments the attempt/claim generation.  The
                # run-level key therefore deduplicates starts and terminal
                # outcomes across all ownership attempts.
                event_key=stable_event_key(claim.id, event_type),
                attributes=attributes,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # Telemetry is informational.  A broken event sink must not turn a
            # durable terminal Run into a retry or a duplicate execution.
            return


def _run_event_type(status: RunStatus) -> str:
    return {
        RunStatus.COMPLETED: "RUN_COMPLETED",
        RunStatus.FAILED: "RUN_FAILED",
        RunStatus.CANCELLED: "RUN_CANCELLED",
    }[status]


def _public_answer_projection(
    answer: dict[str, object],
    result: dict[str, object],
    *,
    runtime_config_snapshot_id: str,
    require_audited: bool,
) -> dict[str, object] | None:
    """Persist only the reviewed public projection of a graph answer."""
    evidence = result.get("evidence")
    parent_ids: list[str] = []
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            parent_id = item.get("parent_id")
            if isinstance(parent_id, str) and parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)
    route_value = result.get("route")
    if isinstance(route_value, dict):
        route_value = route_value.get("route")
    route = route_value if isinstance(route_value, str) else None
    projected = project_public_answer(
        answer,
        evidence_parent_ids=parent_ids if answer.get("audited") is True else [],
        route=route,
        runtime_config_snapshot_id=(runtime_config_snapshot_id if require_audited else None),
        require_audited=require_audited and (route != "chat" or answer.get("audited") is True or answer.get("tool_audited") is True),
    )
    if projected is None:
        return None
    public = projected.model_dump(mode="json", exclude_none=True)
    if not projected.segments:
        public.pop("segments", None)
    if not projected.evidence_parent_ids:
        public.pop("evidence_parent_ids", None)
    return public
