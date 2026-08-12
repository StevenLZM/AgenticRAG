"""Bounded, isolated research subagents and deterministic evidence reduction."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from agentic_rag.query.evidence_builder import EvidenceItem, EvidenceManifestEntry, PackedEvidence
from agentic_rag.query.todos import TodoItem
from agentic_rag.query.tools import ResearchContext, ResearchToolset
from agentic_rag.runtime.concurrency import ConcurrencyManager


class UnresolvedDependencyError(ValueError):
    """Raised before dispatch when a requested Todo cannot run independently."""


@dataclass(frozen=True, slots=True)
class ChildResearchState:
    """A per-invocation, read-only view supplied to exactly one child worker."""

    todo_id: str
    question: str
    scope: Mapping[str, object]
    retrieval_filter: Mapping[str, object]
    memory_summary: str
    evidence_manifest: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class SubagentResult:
    todo_id: str
    evidence: PackedEvidence


@dataclass(frozen=True, slots=True)
class DelegationResult:
    results: tuple[SubagentResult, ...]
    blocked_todo_ids: tuple[str, ...]
    child_states: tuple[ChildResearchState, ...]


class ChildWorker(Protocol):
    async def __call__(self, state: ChildResearchState, tools: "SubagentTools") -> PackedEvidence: ...


class SubagentTools:
    """The only child tool surface: retrieval packing and calculator arithmetic."""

    def __init__(self, tools: ResearchToolset) -> None:
        self._tools = tools

    async def retrieve_evidence(self, *, query: str, context: ResearchContext, target_id: str) -> PackedEvidence:
        _batch, packed = await self._tools.retrieve_evidence(query=query, ctx=context, target_id=target_id)
        return packed

    async def calculator(self, expression: str) -> dict[str, object]:
        return await self._tools.calculator(expression)


class SubagentDispatcher:
    """Dispatch independent Todo work without allowing child recursion or answers."""

    def __init__(
        self,
        *,
        tools: ResearchToolset | object,
        concurrency: ConcurrencyManager,
        worker: ChildWorker,
        memory_summary: str = "",
        evidence_manifest: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self._tools = SubagentTools(tools) if isinstance(tools, ResearchToolset) else tools
        self._concurrency = concurrency
        self._worker = worker
        self._memory_summary = memory_summary
        self._evidence_manifest = _freeze_manifest(evidence_manifest or {})

    async def delegate(
        self,
        items: Sequence[TodoItem],
        context: ResearchContext,
        *,
        max_parallel: int = 3,
        timeout_seconds: float = 20,
        resolved_todo_ids: frozenset[str] = frozenset(),
    ) -> DelegationResult:
        if max_parallel < 1 or timeout_seconds <= 0:
            raise ValueError("max_parallel and timeout_seconds must be positive")
        requested = tuple(items)
        _validate_independent(requested, resolved_todo_ids=resolved_todo_ids)
        child_states = tuple(self._child_state(item, context) for item in requested)
        per_run = self._concurrency.new_subagent_semaphore(max_parallel)

        async def execute(child: ChildResearchState) -> SubagentResult:
            async with self._concurrency.subagent_slot(per_run):
                evidence = await self._worker(child, self._tools)  # type: ignore[arg-type]
                return SubagentResult(todo_id=child.todo_id, evidence=evidence)

        tasks = {asyncio.create_task(execute(child)): child.todo_id for child in child_states}
        done: set[asyncio.Task[SubagentResult]] = set()
        pending: set[asyncio.Task[SubagentResult]] = set(tasks)
        try:
            done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
            completed: list[SubagentResult] = []
            for task in done:
                completed.append(task.result())
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            return DelegationResult(
                results=tuple(sorted(completed, key=lambda result: result.todo_id)),
                blocked_todo_ids=tuple(sorted(tasks[task] for task in pending)),
                child_states=child_states,
            )
        except asyncio.CancelledError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def _child_state(self, item: TodoItem, context: ResearchContext) -> ChildResearchState:
        scope = _freeze_mapping(context.scope.model_dump(mode="json"))
        return ChildResearchState(
            todo_id=item.id,
            question=item.title,
            scope=scope,
            retrieval_filter=scope,
            memory_summary=self._memory_summary,
            evidence_manifest=self._evidence_manifest,
        )


class EvidenceReducer:
    """Merge server-derived evidence without permitting provenance replacement."""

    @staticmethod
    def merge(results: Sequence[SubagentResult]) -> PackedEvidence:
        selected: dict[str, tuple[EvidenceItem, EvidenceManifestEntry]] = {}
        index_generation = ""
        for result in sorted(results, key=lambda value: value.todo_id):
            if not index_generation:
                index_generation = result.evidence.index_generation
            for item in sorted(result.evidence.items, key=lambda value: value.evidence_id):
                manifest = result.evidence.manifest.get(item.evidence_id)
                if item.evidence_id in selected or not _matches_manifest(item, manifest):
                    continue
                assert manifest is not None
                selected[item.evidence_id] = (item, manifest)
        ordered = tuple(selected[key][0] for key in sorted(selected))
        merged_manifest = {key: selected[key][1] for key in sorted(selected)}
        rendered = "\n".join(f"[evidence:{item.evidence_id}]\n{item.content}" for item in ordered)
        return PackedEvidence(
            items=ordered,
            manifest=merged_manifest,
            rendered_context=rendered,
            token_count=len(rendered),
            index_generation=index_generation,
        )


def _validate_independent(items: Sequence[TodoItem], *, resolved_todo_ids: frozenset[str]) -> None:
    ids = {item.id for item in items}
    if len(ids) != len(items):
        raise ValueError("duplicate delegated todo id")
    for item in items:
        if item.status not in {"pending", "in_progress"}:
            raise UnresolvedDependencyError("only unfinished todos can be delegated")
        if not set(item.dependencies).issubset(resolved_todo_ids):
            raise UnresolvedDependencyError("delegated todo has unresolved dependencies")


def _matches_manifest(item: EvidenceItem, manifest: EvidenceManifestEntry | None) -> bool:
    return manifest is not None and (
        manifest.evidence_id,
        manifest.parent_id,
        manifest.document_id,
        manifest.document_version_id,
        manifest.ast_locator,
    ) == (
        item.evidence_id,
        item.parent_id,
        item.document_id,
        item.document_version_id,
        item.ast_locator,
    )


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, object]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_manifest(value: Mapping[str, Mapping[str, object]]) -> Mapping[str, Mapping[str, object]]:
    return MappingProxyType({str(key): _freeze_mapping(item) for key, item in value.items()})
