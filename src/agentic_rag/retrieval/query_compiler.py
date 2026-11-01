"""Compile trusted business meaning without granting document access."""

import re

from agentic_rag.domain.models import UserScope
from agentic_rag.models.contextual import ContextArtifactRef
from agentic_rag.persistence.artifacts import ArtifactRef, ArtifactStore
from agentic_rag.retrieval.filters import FilterBuilder, intersect_documents
from agentic_rag.retrieval.models import QueryPlan, RetrievalRequest
from agentic_rag.retrieval.query_context import (
    BusinessScopeCoverageProvider,
    QueryBusinessContext,
    ScopeResolution,
    TaskBusinessObjective,
    TrustedBusinessEntityDirectory,
    TrustedBusinessProfileProvider,
)
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class BusinessScopeConflict(ValueError):
    """Generated search changed the run's assigned business target."""


def _literal_mentions(text: str) -> tuple[ScopeResolution, ...]:
    # Deliberately finite grammar: exact suffixed company names, never alias guesses.
    pattern = r"(?:(?<=^)|(?<=[，。；、\s与和]))(?:查询|比较|请问|关于)?(?P<company>[\u4e00-\u9fffA-Za-z0-9]{2,20}?公司)"
    found = []
    for match in re.finditer(pattern, text):
        value = match.group("company")
        if value in {"我们公司", "我的公司", "本公司", "其他公司", "哪个公司"}:
            continue
        found.append(
            ScopeResolution(
                field="company",
                value=value,
                source_kind="explicit_user",
                source_ref="user-question",
                source_span=match.span("company"),
                resolution_status="unresolved"
                if re.search(r"[两多各几]家公司$", value)
                else "resolved",
            )
        )
    for match in re.finditer(r"普通员工|管理人员|外包人员", text):
        found.append(
            ScopeResolution(
                field="population",
                value=match.group(),
                source_kind="explicit_user",
                source_ref="user-question",
                source_span=match.span(),
            )
        )
    return tuple(found)


class QueryCompiler:
    def __init__(
        self,
        *,
        profiles: TrustedBusinessProfileProvider | None = None,
        directory: TrustedBusinessEntityDirectory | None = None,
        coverage: BusinessScopeCoverageProvider | None = None,
        artifacts: ArtifactStore | None = None,
        filter_builder: FilterBuilder | None = None,
    ) -> None:
        self.profiles, self.directory, self.coverage, self.artifacts = (
            profiles,
            directory,
            coverage,
            artifacts,
        )
        self.filter_builder = filter_builder or FilterBuilder()

    async def mentions(self, text: str) -> tuple[ScopeResolution, ...]:
        items = (
            await self.directory.resolve_mentions(text)
            if self.directory
            else _literal_mentions(text)
        )
        for item in items:
            if item.source_span is None or text[slice(*item.source_span)] != item.value:
                raise ValueError(
                    "entity resolution must retain exact input source span"
                )
        return items

    async def freeze(
        self, question: str, scope: UserScope, *, task_intent_ref: str = "user-question"
    ) -> QueryBusinessContext:
        profile = await self.profiles.get_profile(scope) if self.profiles else None
        if profile is not None and profile.owner_id != scope.user_id:
            raise ValueError("profile owner mismatch")
        return QueryBusinessContext(
            owner_id=scope.user_id,
            source_question=question,
            task_intent_ref=task_intent_ref,
            trusted_profile=profile,
            source_scope=await self.mentions(question),
        )

    @staticmethod
    def _meaning(
        business: QueryBusinessContext,
    ) -> tuple[tuple[ScopeResolution, ...], str]:
        explicit_fields = {r.field for r in business.source_scope}
        resolved = business.source_scope + tuple(
            r for r in business.confirmed_reference if r.field not in explicit_fields
        )
        if business.task_targets:
            if any(
                t.value
                not in {r.value for r in business.source_scope if r.field == t.field}
                for t in business.task_targets
                if any(r.field == t.field for r in business.source_scope)
            ):
                raise BusinessScopeConflict(
                    "task target conflicts with explicit user target"
                )
            fields = {t.field for t in business.task_targets}
            resolved = (
                tuple(r for r in resolved if r.field not in fields)
                + business.task_targets
            )
        status = "resolved"
        self_directed = bool(
            re.search(
                r"我们公司|我的公司|我(?:去|要|出差)|本公司", business.source_question
            )
        )
        if self_directed:
            profile = business.trusted_profile
            companies = {r.value for r in resolved if r.field == "company"}
            if not companies and (profile is None or not profile.companies):
                status = "unresolved"
            elif not companies and len(profile.companies) != 1:
                status = "ambiguous"
            elif (
                profile is not None
                and len(profile.companies) == 1
                and (not companies or companies == {profile.companies[0]})
            ):
                # Own-profile defaults apply only to a still-missing field in
                # a self-directed own-company scope, never another company.
                existing = {r.field for r in resolved}
                defaults = tuple(
                    ScopeResolution(
                        field=field,
                        value=value,
                        source_kind="trusted_profile",
                        source_ref=profile.revision,
                        resolution_status="defaulted",
                        is_default=True,
                    )
                    for field, value in (
                        ("company", profile.companies[0]),
                        ("population", profile.population),
                    )
                    if value and field not in existing
                )
                resolved += defaults
                if defaults:
                    status = "defaulted"
        return resolved, status

    @classmethod
    def for_task(
        cls,
        business: QueryBusinessContext,
        target: str,
        *,
        task_ref: str,
        document_ids: tuple[str, ...] = (),
    ) -> QueryBusinessContext:
        """Append task intent; common compilation validates its complete ancestry.

        Literal checks provide early rejection only. They cannot authorize or
        narrow directory-only names; the async compiler owns that decision.
        """
        meaning, _ = cls._meaning(business)
        for mention in _literal_mentions(target):
            if mention.resolution_status not in {"resolved", "defaulted"} or not any(
                r.field == mention.field
                and r.value == mention.value
                and r.resolution_status in {"resolved", "defaulted"}
                for r in meaning
            ):
                raise BusinessScopeConflict(
                    "task target requires coordinator reassignment"
                )
        permitted = business.permitted_document_ids
        if document_ids:
            permitted = tuple(
                d for d in document_ids if permitted is None or d in permitted
            )
        return business.model_copy(
            update={
                "task_objectives": (
                    *business.task_objectives,
                    TaskBusinessObjective(text=target, source_ref=task_ref),
                ),
                "task_intent_ref": task_ref,
                "permitted_document_ids": permitted,
            }
        )

    async def compile(
        self,
        request: RetrievalRequest,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
        business: QueryBusinessContext | None,
    ) -> QueryPlan:
        business = (
            business
            if business is not None
            else await self.freeze(request.query, scope)
        )
        if business.owner_id != scope.user_id or (
            business.trusted_profile
            and business.trusted_profile.owner_id != scope.user_id
        ):
            raise ValueError("business context owner mismatch")
        explicit = (
            business.source_scope
            if request.query == business.source_question
            else await self.mentions(request.query)
        )
        resolved, status = self._meaning(business)
        for objective in business.task_objectives:
            mentions = await self.mentions(objective.text)
            for mention in mentions:
                if mention.resolution_status not in {
                    "resolved",
                    "defaulted",
                } or not any(
                    r.field == mention.field
                    and r.value == mention.value
                    and r.resolution_status in {"resolved", "defaulted"}
                    for r in resolved
                ):
                    raise BusinessScopeConflict(
                        "task target requires coordinator reassignment"
                    )
            fields = {r.field for r in mentions}
            narrowed = tuple(
                r.model_copy(
                    update={
                        "source_kind": "task_target",
                        "source_ref": objective.source_ref,
                        "source_span": mention.source_span,
                    }
                )
                for r in resolved
                for mention in mentions
                if (r.field, r.value) == (mention.field, mention.value)
            )
            resolved = tuple(r for r in resolved if r.field not in fields) + narrowed
        # A generated search may narrow an assigned comparison, never invent a target.
        if request.query != business.source_question:
            for item in explicit:
                allowed = {r.value for r in resolved if r.field == item.field}
                if item.value not in allowed or item.resolution_status not in {
                    "resolved",
                    "defaulted",
                }:
                    raise BusinessScopeConflict(
                        "query target requires coordinator reassignment"
                    )
            fields = {r.field for r in explicit}
            selected = {(r.field, r.value) for r in explicit}
            resolved = tuple(
                r
                for r in resolved
                if r.field not in fields or (r.field, r.value) in selected
            )
        if any(r.resolution_status in {"ambiguous", "unresolved"} for r in resolved):
            status = (
                "ambiguous"
                if any(r.resolution_status == "ambiguous" for r in resolved)
                else "unresolved"
            )
        usable = tuple(
            r for r in resolved if r.resolution_status in {"resolved", "defaulted"}
        )
        added = tuple(
            dict.fromkeys(r.value for r in usable if r.value not in request.query)
        )
        prefix = "；".join(added)
        semantic = prefix + "\n\n" + request.query if prefix else request.query
        authorization = self.filter_builder.build(request, scope, snapshot)
        effective = intersect_documents(authorization, request.document_ids or None)
        effective = intersect_documents(effective, business.permitted_document_ids)
        business_filter = None
        binding = None
        companies = {r.value for r in usable if r.field == "company"}
        if self.coverage and companies and status not in {"ambiguous", "unresolved"}:
            cert = await self.coverage.coverage(scope, snapshot)
            current = await self.coverage.current_binding(scope, snapshot)
            if (
                cert
                and cert.complete
                and not cert.unresolved_document_ids
                and not cert.ambiguous_document_ids
                and cert.binding == current
                and cert.owner_id == scope.user_id
                and cert.index_generation == snapshot.index_generation
            ):
                binding = current
                business_filter = tuple(
                    sorted(
                        {
                            doc
                            for doc, company in cert.document_companies
                            if company in companies
                        }
                    )
                )
                effective = intersect_documents(effective, business_filter)
        return QueryPlan(
            original_query=request.query,
            task_intent_ref=business.task_intent_ref,
            profile_revision=business.trusted_profile.revision
            if business.trusted_profile
            else None,
            effective_business_scope=resolved,
            dense_query=semantic,
            lexical_core_query=request.query,
            lexical_scope_terms=added,
            rerank_query=semantic,
            protected_query_context=prefix,
            allow_query_windows=not any(r.value in request.query for r in resolved),
            authorization_filter=authorization,
            business_filter=business_filter,
            effective_filter=effective,
            coverage_binding=binding,
            snapshot_id=snapshot.snapshot_id,
            resolution_status=status,
        )

    def persist(self, plan: QueryPlan) -> ContextArtifactRef | None:
        if self.artifacts is None:
            return None
        ref = self.artifacts.put_json(
            f"query-plans/{plan.plan_hash}.json", plan.model_dump(mode="json")
        )
        return ContextArtifactRef(
            uri=ref.uri, sha256=ref.sha256, size_bytes=ref.size_bytes
        )

    def load(
        self,
        ref: ContextArtifactRef,
        *,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
    ) -> QueryPlan:
        if self.artifacts is None:
            raise ValueError("query plan artifact store missing")
        plan = QueryPlan.model_validate(
            self.artifacts.read_json(ArtifactRef(**ref.model_dump()))
        )
        if (
            plan.effective_filter.user_id != scope.user_id
            or plan.snapshot_id != snapshot.snapshot_id
        ):
            raise ValueError("query plan owner or snapshot mismatch")
        return plan
