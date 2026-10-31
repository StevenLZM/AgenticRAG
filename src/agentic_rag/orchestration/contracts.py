"""Bounded delegation contracts; model suggestions never confer authority."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from agentic_rag.query.evidence_builder import PackedEvidence
from agentic_rag.retrieval.models import RankedCandidates

Key = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9_.:-]{1,80}$")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
Role = Literal["retrieval", "analysis"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Criterion(Contract):
    id: Key
    description: Text
    required: bool = True


class OutputContract(Contract):
    schema_name: Literal["retrieval.v1", "analysis.v1"]
    max_items: int = Field(default=16, ge=1, le=32, description="Maximum individual findings, including one executor-generated finding per calculation. This counts findings plus calculations, not reports or companies. Prefer the default unless an explicitly smaller output is required.")


class ToolGuidance(Contract):
    allowed: tuple[Literal["retrieve", "read_evidence", "calculate"], ...] = Field(min_length=1, max_length=3)
    instructions: Text


class SourceGuidance(Contract):
    instructions: Text
    document_ids: tuple[Key, ...] = Field(default=(), max_length=64)
    preferred_document_ids: tuple[Key, ...] = Field(default=(), max_length=64)
    required_document_ids: tuple[Key, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def bounded_preferences(self) -> "SourceGuidance":
        if self.document_ids and not set((*self.preferred_document_ids, *self.required_document_ids)) <= set(self.document_ids):
            raise ValueError("source document preferences must stay inside hard document scope")
        return self


class TaskBoundary(Contract):
    in_scope: Text
    excluded_scope: tuple[Text, ...] = Field(max_length=16)
    stop_conditions: tuple[Text, ...] = Field(min_length=1, max_length=16)
    escalate_when: tuple[Text, ...] = Field(min_length=1, max_length=16)


class TaskBudget(Contract):
    max_queries: int = Field(default=3, ge=1, le=8)
    max_rounds: int = Field(default=3, ge=1, le=8)
    max_no_progress_rounds: int = Field(default=2, ge=1, le=4)
    max_model_calls: int = Field(default=32, ge=1, le=128, description="Cumulative provider calls across all task rounds, including retries and repairs.")
    max_output_tokens: int = Field(default=262144, ge=128, le=1048576, description="Cumulative output tokens across all task rounds and provider attempts, not a per-call limit. The default accommodates a 100000-token analysis plus one repair and assessment.")


class TaskDependency(Contract):
    task_key: Key
    allow_partial: bool = False
    criterion_ids: tuple[Key, ...] = Field(default=(), max_length=32)


class TaskSpec(Contract):
    key: Key
    agent_type: Role
    objective: Text
    criteria: tuple[Criterion, ...] = Field(min_length=1, max_length=32)
    output: OutputContract
    tools: ToolGuidance
    sources: SourceGuidance
    boundary: TaskBoundary
    dependencies: tuple[TaskDependency, ...] = Field(default=(), max_length=16)
    budget: TaskBudget = Field(default_factory=TaskBudget)
    required_for_answer: bool = True

    @model_validator(mode="after")
    def role_boundary(self) -> "TaskSpec":
        allowed = {"retrieve"} if self.agent_type == "retrieval" else {"read_evidence", "calculate"}
        if not set(self.tools.allowed) <= allowed:
            raise ValueError("tools exceed the server-owned agent role")
        if self.output.schema_name != f"{self.agent_type}.v1":
            raise ValueError("output schema does not match agent role")
        if len({c.id for c in self.criteria}) != len(self.criteria):
            raise ValueError("criterion IDs must be unique within a task")
        if len({d.task_key for d in self.dependencies}) != len(self.dependencies):
            raise ValueError("duplicate dependency")
        return self


class GroupMember(Contract):
    task_key: Key
    required: bool = True


class RetrievalGroupSpec(Contract):
    key: Key
    objective: Text
    criteria: tuple[Criterion, ...] = Field(min_length=1, max_length=32)
    members: tuple[GroupMember, ...] = Field(min_length=1, max_length=16)
    join_timeout_seconds: int = Field(default=20, ge=1, le=120)
    max_aggregations: int = Field(default=3, ge=1, le=8)
    retained_fast_criteria: tuple[Key, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def required_members(self) -> "RetrievalGroupSpec":
        if not any(m.required for m in self.members):
            raise ValueError("a retrieval group needs at least one required member")
        if len({m.task_key for m in self.members}) != len(self.members):
            raise ValueError("duplicate group member")
        if len({c.id for c in self.criteria}) != len(self.criteria):
            raise ValueError("duplicate group criterion")
        if not set(self.retained_fast_criteria) <= {c.id for c in self.criteria}:
            raise ValueError("retained fast evidence must map to current group criteria")
        return self


class DispatchPlan(Contract):
    tasks: tuple[TaskSpec, ...] = Field(min_length=1, max_length=16)
    groups: tuple[RetrievalGroupSpec, ...] = Field(default=(), max_length=16)


class Gap(Contract):
    criterion_id: Key
    description: Text
    related_task_keys: tuple[Key, ...] = ()
    evidence_refs: tuple[Key, ...] = ()
    search_direction: str = Field(default="", max_length=2000)


class Finding(Contract):
    statement: Text
    evidence_refs: tuple[Key, ...] = Field(min_length=1, max_length=32, description="Required source IDs supporting this finding; use query for grounded user-input analysis, never leave empty.")
    criterion_ids: tuple[Key, ...] = Field(default=(), max_length=32)
    kind: Literal["fact", "calculation", "inference"] = "fact"
    calculation_refs: tuple[Key, ...] = Field(default=(), max_length=16)
    limitations: tuple[Text, ...] = Field(default=(), max_length=16)


class CandidateResult(Contract):
    origin_key: Key
    origin: Literal["agent_task", "fast_pipeline"] = "agent_task"
    task_revision: int = Field(default=1, ge=1)
    input_version: int = Field(default=1, ge=1)
    round_id: int = Field(default=1, ge=1)
    criterion_ids: tuple[Key, ...] = Field(min_length=1, max_length=32)
    candidates: tuple[RankedCandidates, ...] = Field(min_length=1, max_length=8)
    findings: tuple[Finding, ...] = Field(default=(), max_length=32)
    gaps: tuple[Gap, ...] = Field(default=(), max_length=32)
    queries: tuple[str, ...] = Field(default=(), max_length=32)
    context: dict[str, Any] = Field(default_factory=dict)


class PreparedBundle(Contract):
    packed: PackedEvidence
    input_hash: str
    criteria: tuple[Criterion, ...]
    source_map: dict[str, tuple[str, ...]]
    candidate_map: dict[str, tuple[str, ...]]
    uncovered_criteria: tuple[str, ...] = ()

    @property
    def bundle_hash(self) -> str:
        return content_hash(self)


class CriterionCoverage(Contract):
    criterion_id: Key
    status: Literal["supported", "unsupported", "conflict"]
    evidence_refs: tuple[Key, ...] = Field(default=(), max_length=32)
    explanation: str = Field(default="", max_length=2000)


class GradeResult(Contract):
    bundle_hash: str
    decision: Literal["sufficient", "insufficient", "conflict", "clarification_required"]
    coverage: tuple[CriterionCoverage, ...] = Field(max_length=32)
    findings: tuple[Finding, ...] = Field(default=(), max_length=32)
    gaps: tuple[Gap, ...] = Field(default=(), max_length=32)


def content_hash(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
