"""Bounded, isolated research subagents and deterministic evidence reduction."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from agentic_rag.query.evidence_builder import (
    EvidenceItem,
    EvidenceManifestEntry,
    PackedEvidence,
)
from agentic_rag.query.todos import TodoDependencyInput, TodoItem
from agentic_rag.query.tools import ResearchContext, ResearchToolset
from agentic_rag.retrieval.models import EvidenceBatch
from agentic_rag.runtime.concurrency import ConcurrencyManager
from agentic_rag.safety.context import DataEnvelope


class UnresolvedDependencyError(ValueError):
    """Raised before dispatch when a requested Todo cannot run independently."""


class EvidenceConsistencyError(ValueError):
    """Raised when child evidence cannot share one immutable index generation."""


@dataclass(frozen=True, slots=True)
class ChildResearchState:
    """A per-invocation, read-only view supplied to exactly one child worker."""

    todo_id: str
    question: str
    scope: Mapping[str, object]
    retrieval_filter: Mapping[str, object]
    memory_summary: str
    evidence_manifest: Mapping[str, Mapping[str, object]]
    dependency_inputs: tuple[TodoDependencyInput, ...] = ()
    dependency_context: str = ""
    dependency_results: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SubagentResult:
    todo_id: str
    evidence: PackedEvidence
    batch: EvidenceBatch | None = None


@dataclass(frozen=True, slots=True)
class DelegationResult:
    results: tuple[SubagentResult, ...]
    blocked_todo_ids: tuple[str, ...]
    child_states: tuple[ChildResearchState, ...]


class ChildWorker(Protocol):
    async def __call__(
        self, state: ChildResearchState, tools: "SubagentTools"
    ) -> PackedEvidence | tuple[EvidenceBatch, PackedEvidence]: ...


class SubagentTools:
    """The only child tool surface: retrieval packing and calculator arithmetic."""

    def __init__(self, tools: ResearchToolset) -> None:
        self._tools = tools

    async def retrieve_evidence(
        self, *, query: str, context: ResearchContext, target_id: str
    ) -> tuple[EvidenceBatch, PackedEvidence]:
        return await self._tools.retrieve_evidence(
            query=query, ctx=context, target_id=target_id
        )

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
        packed_evidence: PackedEvidence | None = None,
        dependency_inputs: Mapping[str, tuple[TodoDependencyInput, ...]] | None = None,
        task_results: object = None,
        memory_summary: str | None = None,
        queries: Mapping[str, str] | None = None,
    ) -> DelegationResult:
        if max_parallel < 1 or timeout_seconds <= 0:
            raise ValueError("max_parallel and timeout_seconds must be positive")
        requested = tuple(items)
        _validate_independent(requested, resolved_todo_ids=resolved_todo_ids)
        child_states = tuple(self._child_state(
            item, context, packed_evidence=packed_evidence,
            dependency_inputs=(dependency_inputs or {}).get(item.id, ()),
            task_results=task_results, memory_summary=memory_summary,
            query=(queries or {}).get(item.id),
        ) for item in requested)
        per_run = self._concurrency.new_subagent_semaphore(max_parallel)

        async def execute(child: ChildResearchState) -> SubagentResult:
            async with self._concurrency.subagent_slot(per_run):
                result = await self._worker(child, self._tools)  # type: ignore[arg-type]
                if isinstance(result, tuple) and len(result) == 2:
                    batch, evidence = result
                    if not isinstance(batch, EvidenceBatch) or not isinstance(
                        evidence, PackedEvidence
                    ):
                        raise TypeError("child worker returned malformed evidence pair")
                    return SubagentResult(
                        todo_id=child.todo_id, evidence=evidence, batch=batch
                    )
                if isinstance(result, PackedEvidence):
                    # Compatibility for existing workers while the raw-batch
                    # contract rolls out. New workers should return a pair so
                    # the supervisor can rebuild one global pack.
                    return SubagentResult(todo_id=child.todo_id, evidence=result)
                raise TypeError("child worker returned malformed evidence")

        tasks = {asyncio.create_task(execute(child)): child.todo_id for child in child_states}
        done: set[asyncio.Task[SubagentResult]] = set()
        pending: set[asyncio.Task[SubagentResult]] = set(tasks)
        try:
            done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
            completed: list[SubagentResult] = []
            failed: list[str] = []
            for task in done:
                try:
                    completed.append(task.result())
                except (TypeError, ValueError):
                    # Provenance/schema corruption is terminal, unlike an unavailable child.
                    raise
                except Exception:
                    failed.append(tasks[task])
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            return DelegationResult(
                results=tuple(sorted(completed, key=lambda result: result.todo_id)),
                blocked_todo_ids=tuple(sorted([*(tasks[task] for task in pending), *failed])),
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

    def _child_state(
        self, item: TodoItem, context: ResearchContext, *,
        packed_evidence: PackedEvidence | None,
        dependency_inputs: tuple[TodoDependencyInput, ...], task_results: object,
        memory_summary: str | None, query: str | None,
    ) -> ChildResearchState:
        scope = _freeze_mapping(context.scope.model_dump(mode="json"))
        dependency_text = []
        dependency_results: dict[str, object] = {}
        manifest = self._evidence_manifest
        if packed_evidence is not None:
            if packed_evidence.index_generation != context.snapshot.index_generation:
                raise EvidenceConsistencyError("dependency generation mismatch")
            manifest = _freeze_manifest({key: value.model_dump(mode="json")
                                         for key, value in packed_evidence.manifest.items()})
            references = {key for dep in dependency_inputs for key in dep.evidence_ids}
            if not references.issubset(manifest):
                raise EvidenceConsistencyError("dependency evidence missing")
            for evidence in packed_evidence.items:
                if evidence.evidence_id not in references:
                    continue
                if not _matches_manifest(evidence, packed_evidence.manifest.get(evidence.evidence_id)):
                    raise EvidenceConsistencyError("dependency manifest mismatch")
                dependency_text.append(DataEnvelope(
                    source_label=f"document:{evidence.document_id}", evidence_id=evidence.evidence_id,
                    content=evidence.content, heading_path=evidence.heading_path,
                ).render())
        for dep in dependency_inputs:
            if dep.todo_id not in item.blocked_by:
                raise EvidenceConsistencyError("unexpected dependency input")
            if dep.result_ref:
                if not isinstance(task_results, Mapping) or dep.result_ref not in task_results:
                    raise EvidenceConsistencyError("dependency result missing")
                dependency_results[dep.result_ref] = task_results[dep.result_ref]
        return ChildResearchState(
            todo_id=item.id,
            question=query or item.title,
            scope=scope,
            retrieval_filter=scope,
            memory_summary=self._memory_summary if memory_summary is None else memory_summary,
            evidence_manifest=manifest,
            dependency_inputs=dependency_inputs,
            dependency_context="\n".join(dependency_text),
            dependency_results=_freeze_mapping(dependency_results),
        )


class EvidenceReducer:
    """Merge server-derived evidence without permitting provenance replacement."""

    @staticmethod
    def merge(
        results: Sequence[SubagentResult],
        *,
        expected_index_generation: str | None = None,
    ) -> PackedEvidence:
        selected: dict[str, tuple[EvidenceItem, EvidenceManifestEntry]] = {}
        generations = {result.evidence.index_generation for result in results}
        if not generations or "" in generations:
            raise EvidenceConsistencyError("evidence index generation must be non-empty")
        if len(generations) != 1:
            raise EvidenceConsistencyError("evidence index generation must be consistent")
        index_generation = next(iter(generations))
        if expected_index_generation is not None and index_generation != expected_index_generation:
            raise EvidenceConsistencyError("evidence does not match expected index generation")
        for result in sorted(results, key=lambda value: value.todo_id):
            for item in sorted(result.evidence.items, key=lambda value: value.evidence_id):
                manifest = result.evidence.manifest.get(item.evidence_id)
                if item.evidence_id in selected or not _matches_manifest(item, manifest):
                    continue
                assert manifest is not None
                selected[item.evidence_id] = (item, manifest)
        ordered = tuple(selected[key][0] for key in sorted(selected))
        merged_manifest = {key: selected[key][1] for key in sorted(selected)}
        rendered = "\n".join(
            DataEnvelope(
                source_label=f"document:{item.document_id}",
                evidence_id=item.evidence_id,
                content=item.content,
                heading_path=item.heading_path,
            ).render()
            for item in ordered
        )
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
        manifest.heading_path,
    ) == (
        item.evidence_id,
        item.parent_id,
        item.document_id,
        item.document_version_id,
        item.ast_locator,
        item.heading_path,
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
