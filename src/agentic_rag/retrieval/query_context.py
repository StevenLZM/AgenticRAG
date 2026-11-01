"""Server-owned business meaning, separate from access authorization."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.domain.models import UserScope
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class FrozenContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ScopeResolution(FrozenContract):
    field: Literal["company", "population", "period", "unit"]
    value: str
    source_kind: Literal[
        "explicit_user", "task_target", "confirmed_reference", "trusted_profile"
    ]
    source_ref: str
    source_span: tuple[int, int] | None = None
    resolution_status: Literal["resolved", "defaulted", "ambiguous", "unresolved"] = (
        "resolved"
    )
    is_default: bool = False
    allowed_uses: tuple[Literal["semantic"], ...] = ("semantic",)
    resolver_version: str = "literal-v1"


class TrustedBusinessProfile(FrozenContract):
    owner_id: str
    revision: str
    companies: tuple[str, ...] = ()
    population: str | None = None


class TaskBusinessObjective(FrozenContract):
    """Pending task intent; resolved against frozen assignment before recall."""

    text: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)


class QueryBusinessContext(FrozenContract):
    owner_id: str
    source_question: str
    task_intent_ref: str = "user-question"
    trusted_profile: TrustedBusinessProfile | None = None
    source_scope: tuple[ScopeResolution, ...] = ()
    task_targets: tuple[ScopeResolution, ...] = ()
    task_objectives: tuple[TaskBusinessObjective, ...] = ()
    confirmed_reference: tuple[ScopeResolution, ...] = ()
    permitted_document_ids: tuple[str, ...] | None = None


class CoverageBinding(FrozenContract):
    owner_id: str
    corpus_snapshot: str
    index_generation: str
    version_set_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    mapping_revision: str
    issuer: str
    rule_version: str


class BusinessScopeCoverage(CoverageBinding):
    complete: bool
    unresolved_document_ids: tuple[str, ...] = ()
    ambiguous_document_ids: tuple[str, ...] = ()
    document_companies: tuple[tuple[str, str], ...] = ()

    @property
    def binding(self) -> CoverageBinding:
        return CoverageBinding.model_validate(
            self.model_dump(include=set(CoverageBinding.model_fields))
        )


class TrustedBusinessProfileProvider(Protocol):
    async def get_profile(self, scope: UserScope) -> TrustedBusinessProfile | None: ...


class TrustedBusinessEntityDirectory(Protocol):
    """Resolve literal mentions; ambiguous aliases must retain ambiguous status."""

    async def resolve_mentions(self, text: str) -> tuple[ScopeResolution, ...]: ...


class BusinessScopeCoverageProvider(Protocol):
    async def coverage(
        self, scope: UserScope, snapshot: RuntimeConfigSnapshot
    ) -> BusinessScopeCoverage | None: ...
    async def current_binding(
        self, scope: UserScope, snapshot: RuntimeConfigSnapshot
    ) -> CoverageBinding: ...


def business_kwargs(context: QueryBusinessContext | None) -> dict:
    """Omit the new keyword for legacy compositions and their existing ports."""
    return {"business_context": context} if context is not None else {}
