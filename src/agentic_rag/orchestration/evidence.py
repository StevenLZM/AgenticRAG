"""External fan-in preparation: authorize, hydrate, deduplicate and pack."""

from collections.abc import Sequence

from agentic_rag.domain.models import UserScope
from agentic_rag.orchestration.contracts import CandidateResult, Criterion, PreparedBundle, content_hash
from agentic_rag.query.evidence_builder import EvidenceBuilder, EvidenceCoverageTarget
from agentic_rag.retrieval.graph import ParentHydrator
from agentic_rag.retrieval.models import EvidenceBatch, ParentEvidence, RankedCandidates
from agentic_rag.retrieval.parents import aggregate_parents
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class EvidencePreparer:
    """Consumes a frozen candidate set; it does not decide factual sufficiency."""

    def __init__(self, hydrator: ParentHydrator, builder: EvidenceBuilder | None = None) -> None:
        self.hydrator = hydrator
        self.builder = builder or EvidenceBuilder()

    async def prepare(
        self, candidates: Sequence[CandidateResult], criteria: Sequence[Criterion],
        scope: UserScope, snapshot: RuntimeConfigSnapshot, *, coverage_aliases: dict[str, tuple[str, ...]] | None = None,
    ) -> PreparedBundle:
        parents: dict[tuple[str, str], ParentEvidence] = {}
        sources: dict[tuple[str, str], list[str]] = {}
        selections: list[tuple[CandidateResult, RankedCandidates, list[ParentEvidence]]] = []
        children: dict[str, tuple[str, str, str, str]] = {}
        for result in candidates:
            for batch in result.candidates:
                if batch.user_id != scope.user_id:
                    raise ValueError("candidate tenant scope mismatch")
                if batch.index_generation != snapshot.index_generation or batch.snapshot_id != snapshot.snapshot_id:
                    raise ValueError("candidate snapshot or index generation mismatch")
                for hit in batch.hits:
                    if hit.user_id != scope.user_id or (batch.document_ids and hit.document_id not in batch.document_ids):
                        raise ValueError("candidate source is outside scope")
                    identity = (hit.parent_id, hit.document_id, hit.document_version_id, hit.content)
                    if hit.child_id in children and children[hit.child_id] != identity:
                        raise ValueError("conflicting candidate identity")
                    children[hit.child_id] = identity
                budget = snapshot.evaluation.retrieval_budget if snapshot.evaluation else None
                selected = aggregate_parents(batch.hits, limit=budget.parent_k if budget else 6, max_children_per_parent=2)
                selections.append((result, batch, selected))
                for parent in selected:
                    key = (parent.parent_id, parent.document_version_id)
                    if key in parents and parents[key].document_id != parent.document_id:
                        raise ValueError("parent provenance conflict")
                    if key not in parents:
                        parents[key] = parent
                    else:
                        old = parents[key]
                        hits = {h.child_id: h for h in (*old.child_hits, *parent.child_hits)}
                        parents[key] = old.model_copy(update={"child_hits": tuple(hits.values())})
                    origins = sources.setdefault(key, [])
                    if result.origin_key not in origins:
                        origins.append(result.origin_key)
        hydrated = await self.hydrator.hydrate(list(parents.values()), scope)
        by_key = {(p.parent_id, p.document_version_id): p for p in hydrated}
        if set(by_key) != set(parents):
            raise ValueError("hydration did not preserve the authorized parent set")
        batches = [
            EvidenceBatch(
                query=batch.query,
                parents=tuple(by_key[(p.parent_id, p.document_version_id)] for p in selected),
                document_ids=batch.document_ids, target_ids=(coverage_aliases or {}).get(result.origin_key, result.criterion_ids),
                degraded_components=batch.degraded_components,
            )
            for result, batch, selected in selections
        ]
        packed = self.builder.build(
            batches, [EvidenceCoverageTarget(target_id=c.id, description=c.description) for c in criteria],
            scope, snapshot, rank_across_batches=False, balance_targets=True,
        )
        source_map = {
            item.evidence_id: tuple(sources[(item.parent_id, item.document_version_id)])
            for item in packed.items
        }
        candidate_map: dict[str, tuple[str, ...]] = {}
        for item in packed.items:
            parent = by_key[(item.parent_id, item.document_version_id)]
            for hit in parent.child_hits:
                candidate_map[hit.child_id] = (*candidate_map.get(hit.child_id, ()), item.evidence_id)
        covered = {target for item in packed.items for target in item.covered_target_ids}
        return PreparedBundle(
            packed=packed,
            input_hash=content_hash([result.model_dump(mode="json") for result in candidates]),
            criteria=tuple(criteria), source_map=source_map, candidate_map=candidate_map,
            uncovered_criteria=tuple(c.id for c in criteria if c.id not in covered),
        )
