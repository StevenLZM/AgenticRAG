"""Best-effort user-facing phase notifications, independent of graph state."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import uuid4

from agentic_rag.domain.models import UserScope
from agentic_rag.observability.logging import AgentEventEmitter
from agentic_rag.persistence.repositories import AgentEvent
from agentic_rag.runtime.models import RuntimeConfigSnapshot

QueryPhase = Literal["processing", "retrieving", "researching", "auditing"]
PHASES = frozenset({"processing", "retrieving", "researching", "auditing"})
PhaseReporter = Callable[[QueryPhase], Awaitable[None]]


async def report_safely(reporter: PhaseReporter | None, phase: QueryPhase) -> None:
    if reporter is not None:
        try:
            await reporter(phase)
        except Exception:
            # Cancellation and process exit are BaseExceptions and propagate.
            pass


class PhaseEventRecorder(Protocol):
    async def append(self, event: AgentEvent) -> int: ...


class QueryPhaseEmitter:
    def __init__(
        self,
        *,
        run_id: str,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
        event_emitter: AgentEventEmitter | None = None,
        event_repository: PhaseEventRecorder | None = None,
    ):
        self.run_id, self.scope, self.snapshot = run_id, scope, snapshot
        self.emitter, self.repository = event_emitter, event_repository

    async def report(self, phase: QueryPhase) -> None:
        if phase not in PHASES:
            return
        emitter = self.emitter
        if (
            isinstance(emitter, AgentEventEmitter)
            and self.snapshot.evaluation is not None
            and self.snapshot.model_copy(update={"evaluation": None}).snapshot_id
            == emitter.runtime_config_snapshot_id
        ):
            emitter = emitter.with_snapshot(self.snapshot.snapshot_id)
        try:
            if emitter is not None:
                if emitter.runtime_config_snapshot_id != self.snapshot.snapshot_id:
                    return
                await emitter.emit(
                    run_id=self.run_id,
                    user_id=self.scope.user_id,
                    event_type="QUERY_PHASE_CHANGED",
                    summary=phase,
                    node_name="query_phase",
                    event_key=uuid4().hex,
                )
            elif self.repository is not None:
                await self.repository.append(
                    AgentEvent(
                        event_key=uuid4().hex,
                        trace_id=self.run_id,
                        run_id=self.run_id,
                        user_id=self.scope.user_id,
                        event_type="QUERY_PHASE_CHANGED",
                        summary=phase,
                        runtime_config_snapshot_id=self.snapshot.snapshot_id,
                        node_name="query_phase",
                        created_at=datetime.now(UTC),
                    )
                )
        except Exception:
            pass
