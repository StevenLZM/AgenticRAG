"""Tests for bounded, isolated research subagents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.query.evidence_builder import EvidenceItem, EvidenceManifestEntry, PackedEvidence
from agentic_rag.query.todos import TodoItem
from agentic_rag.query.tools import ResearchContext
from agentic_rag.retrieval.models import EvidenceBatch
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test", graph_version="query-v1", prompt_version="prompt-v1",
    main_model_id="main", light_model_id="light", embedding_model="embedding",
    embedding_dimensions=1024, reranker_version="reranker", retrieval_config_version="retrieval",
    index_generation="index", memory_config_version="memory",
)
CONTEXT = ResearchContext(scope=UserScope(user_id="u-1"), snapshot=SNAPSHOT)


def _packed(
    evidence_id: str, content: str = "evidence", *, generation: str = "index"
) -> PackedEvidence:
    item = EvidenceItem(
        evidence_id=evidence_id, parent_id=f"parent-{evidence_id}", document_id="doc-1",
        document_version_id="v1", content=content, ast_locator="#/1", covered_target_ids=("todo",),
    )
    return PackedEvidence(
        items=(item,),
        manifest={evidence_id: EvidenceManifestEntry(
            evidence_id=evidence_id, parent_id=item.parent_id, document_id=item.document_id,
            document_version_id=item.document_version_id, ast_locator=item.ast_locator,
        )},
        rendered_context=f"[{evidence_id}] {content}", token_count=len(content), index_generation=generation,
    )


def _batch_for(evidence_id: str) -> EvidenceBatch:
    """Minimal raw retrieval batch paired with one child pack."""
    del evidence_id
    return EvidenceBatch(query="question", parents=())


def _todo(todo_id: str, *, dependencies: tuple[str, ...] = (), status: str = "pending") -> TodoItem:
    return TodoItem(id=todo_id, title=f"Question {todo_id}", owner="supervisor", dependencies=dependencies, status=status)  # type: ignore[arg-type]


@dataclass
class RecordingWorker:
    delay: float = 0
    active: int = 0
    peak: int = 0
    started: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)

    async def __call__(self, state: object, tools: object) -> PackedEvidence:
        del tools
        todo_id = state.todo_id
        self.started.append(todo_id)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            return _packed(f"e-{todo_id}")
        except asyncio.CancelledError:
            self.cancelled.append(todo_id)
            raise
        finally:
            self.active -= 1


async def test_dispatcher_caps_requested_parallelism_at_per_run_limit() -> None:
    from agentic_rag.query.subagents import SubagentDispatcher
    from agentic_rag.runtime.concurrency import ConcurrencyManager

    worker = RecordingWorker(delay=0.02)
    dispatcher = SubagentDispatcher(
        tools=object(), concurrency=ConcurrencyManager(per_run_subagent_limit=2), worker=worker
    )

    result = await dispatcher.delegate([_todo("a"), _todo("b"), _todo("c")], CONTEXT, max_parallel=99)

    assert worker.peak == 2
    assert tuple(item.todo_id for item in result.results) == ("a", "b", "c")
    assert result.blocked_todo_ids == ()


async def test_dispatcher_keeps_completed_results_and_blocks_timeout_remainder() -> None:
    from agentic_rag.query.subagents import SubagentDispatcher
    from agentic_rag.runtime.concurrency import ConcurrencyManager

    class PartialWorker:
        async def __call__(self, state: object, tools: object) -> PackedEvidence:
            del tools
            if state.todo_id == "slow":
                await asyncio.sleep(1)
            return _packed(f"e-{state.todo_id}")

    dispatcher = SubagentDispatcher(tools=object(), concurrency=ConcurrencyManager(), worker=PartialWorker())
    result = await dispatcher.delegate([_todo("fast"), _todo("slow")], CONTEXT, max_parallel=2, timeout_seconds=0.02)

    assert tuple(item.todo_id for item in result.results) == ("fast",)
    assert result.blocked_todo_ids == ("slow",)


def test_reducer_orders_by_todo_then_evidence_id_and_deduplicates() -> None:
    from agentic_rag.query.subagents import EvidenceReducer, SubagentResult

    merged = EvidenceReducer.merge((
        SubagentResult(todo_id="z", evidence=_packed("e-2")),
        SubagentResult(todo_id="a", evidence=_packed("e-2", "first winner")),
        SubagentResult(todo_id="a", evidence=_packed("e-1")),
    ))

    assert [item.evidence_id for item in merged.items] == ["e-1", "e-2"]
    assert merged.items[1].content == "first winner"
    assert list(merged.manifest) == ["e-1", "e-2"]


def test_reducer_rejects_mixed_index_generations() -> None:
    from agentic_rag.query.subagents import EvidenceConsistencyError, EvidenceReducer, SubagentResult

    with pytest.raises(EvidenceConsistencyError, match="index generation"):
        EvidenceReducer.merge((
            SubagentResult(todo_id="a", evidence=_packed("e-a", generation="index-1")),
            SubagentResult(todo_id="b", evidence=_packed("e-b", generation="index-2")),
        ))


def test_reducer_rejects_generation_mismatch_with_expected_snapshot() -> None:
    from agentic_rag.query.subagents import EvidenceConsistencyError, EvidenceReducer, SubagentResult

    with pytest.raises(EvidenceConsistencyError, match="expected index generation"):
        EvidenceReducer.merge(
            (SubagentResult(todo_id="a", evidence=_packed("e-a")),),
            expected_index_generation="index-other",
        )


async def test_dispatcher_rejects_unresolved_dependencies_without_starting_children() -> None:
    from agentic_rag.query.subagents import UnresolvedDependencyError, SubagentDispatcher
    from agentic_rag.runtime.concurrency import ConcurrencyManager

    worker = RecordingWorker()
    dispatcher = SubagentDispatcher(tools=object(), concurrency=ConcurrencyManager(), worker=worker)

    with pytest.raises(UnresolvedDependencyError):
        await dispatcher.delegate([_todo("child", dependencies=("parent",))], CONTEXT)

    assert worker.started == []


async def test_dispatcher_allows_dependencies_already_completed_by_parent() -> None:
    from agentic_rag.query.subagents import SubagentDispatcher
    from agentic_rag.runtime.concurrency import ConcurrencyManager

    worker = RecordingWorker()
    dispatcher = SubagentDispatcher(tools=object(), concurrency=ConcurrencyManager(), worker=worker)

    result = await dispatcher.delegate(
        [_todo("child", dependencies=("parent",))],
        CONTEXT,
        resolved_todo_ids=frozenset({"parent"}),
    )

    assert tuple(item.todo_id for item in result.results) == ("child",)


async def test_child_state_isolated_and_keeps_server_owned_scope_and_manifest() -> None:
    from agentic_rag.query.subagents import SubagentDispatcher
    from agentic_rag.runtime.concurrency import ConcurrencyManager

    observed: list[object] = []

    async def worker(state: object, tools: object) -> PackedEvidence:
        del tools
        observed.append(state)
        return _packed("e-one")

    dispatcher = SubagentDispatcher(
        tools=object(), concurrency=ConcurrencyManager(), worker=worker,
        memory_summary="read only", evidence_manifest={"old": {"document_id": "d"}},
    )
    result = await dispatcher.delegate([_todo("one")], CONTEXT)

    assert result.child_states[0].scope == {"user_id": "u-1"}
    assert result.child_states[0].memory_summary == "read only"
    assert result.child_states[0].evidence_manifest == {"old": {"document_id": "d"}}
    assert observed[0] is result.child_states[0]


async def test_dispatcher_preserves_raw_retrieval_batch_from_child_worker() -> None:
    from agentic_rag.query.subagents import SubagentDispatcher
    from agentic_rag.runtime.concurrency import ConcurrencyManager

    class Worker:
        async def __call__(self, state: object, tools: object) -> tuple[EvidenceBatch, PackedEvidence]:
            del tools
            return _batch_for(state.todo_id), _packed(f"e-{state.todo_id}")

    dispatcher = SubagentDispatcher(
        tools=object(), concurrency=ConcurrencyManager(), worker=Worker()
    )

    result = await dispatcher.delegate([_todo("one")], CONTEXT)

    assert result.results[0].batch is not None
    assert result.results[0].batch.query == "question"


async def test_parent_cancellation_cleans_up_children() -> None:
    from agentic_rag.query.subagents import SubagentDispatcher
    from agentic_rag.runtime.concurrency import ConcurrencyManager

    worker = RecordingWorker(delay=5)
    dispatcher = SubagentDispatcher(tools=object(), concurrency=ConcurrencyManager(), worker=worker)
    task = asyncio.create_task(dispatcher.delegate([_todo("a"), _todo("b")], CONTEXT, max_parallel=2))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(worker.cancelled) == ["a", "b"]
    assert worker.active == 0
