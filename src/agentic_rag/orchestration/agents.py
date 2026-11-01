"""Isolated specialist stages. Tools and source authority are server-owned."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeVar

from pydantic import Field, model_validator

from agentic_rag.domain.models import UserScope
from agentic_rag.orchestration.assessment import validate_grade
from agentic_rag.orchestration.numeric_grounding import evidence_text, unsupported_numbers
from agentic_rag.orchestration.contracts import (
    CandidateResult, Contract, Finding, Gap, GradeResult, Key, PreparedBundle, TaskSpec, Text, content_hash,
)
from agentic_rag.query.calculator import SafeCalculator
from agentic_rag.retrieval.models import RankedCandidates, RetrievalRequest
from agentic_rag.retrieval.query_context import QueryBusinessContext, business_kwargs
from agentic_rag.retrieval.query_compiler import QueryCompiler
from agentic_rag.runtime.models import RuntimeConfigSnapshot

T = TypeVar("T", bound=Contract)


class Execution(Protocol):
    scope: UserScope
    snapshot: RuntimeConfigSnapshot
    question: str
    business_context: QueryBusinessContext | None

    async def structured(self, role: str, payload: dict[str, Any], schema: type[T], *, light: bool = False) -> T: ...
    async def debit(self, kind: str, amount: int = 1) -> None: ...


class Retriever(Protocol):
    async def retrieve_candidates(self, request: RetrievalRequest, scope: UserScope, snapshot: RuntimeConfigSnapshot, *, business_context: QueryBusinessContext | None = None) -> RankedCandidates: ...


class QueryPlan(Contract):
    queries: tuple[Text, ...] = Field(min_length=1, max_length=3)
    reason: Text


def query_schema(max_queries: int) -> type[QueryPlan]:
    class BoundedQueryPlan(QueryPlan):
        queries: tuple[Text, ...] = Field(min_length=1, max_length=max_queries)
    return BoundedQueryPlan


class CandidateSummary(Contract):
    findings: tuple[Finding, ...] = Field(default=(), max_length=32)
    gaps: tuple[Gap, ...] = Field(default=(), max_length=32)


def validate_findings(findings: Sequence[Finding], allowed: set[str], criteria: set[str], *, calculations: set[str] | None = None) -> None:
    for finding in findings:
        if not finding.evidence_refs or not set(finding.evidence_refs) <= allowed:
            raise ValueError("finding source is not in the permitted evidence")
        if not set(finding.criterion_ids) <= criteria:
            raise ValueError("finding criterion is outside task scope")
        if finding.kind == "calculation" or finding.calculation_refs:
            if calculations is None or not finding.calculation_refs or not set(finding.calculation_refs) <= calculations:
                raise ValueError("calculation requires an analysis-owned verified computation")


class RetrievalAgent:
    def __init__(self, retrieval: Retriever) -> None:
        self.retrieval = retrieval

    async def execute(
        self, task: TaskSpec, context: dict[str, Any], execution: Execution, *,
        task_revision: int = 1, input_version: int = 1, round_id: int = 1,
        gaps: Sequence[Gap] = (),
    ) -> CandidateResult:
        if task.agent_type != "retrieval" or set(task.tools.allowed) != {"retrieve"}:
            raise ValueError("retrieval role cannot invoke other tools")
        prior = context.get("queries", [])
        remaining = task.budget.max_queries - len(prior)
        if remaining <= 0 or round_id > task.budget.max_rounds:
            raise ValueError("retrieval query/round budget exhausted")
        rounds = task.budget.max_rounds - round_id + 1
        per_round = min(3, (remaining + rounds - 1) // rounds)
        plan = await execution.structured("retrieval_plan", {
            "task": task.model_dump(mode="json"), "context": context,
            "gaps": [g.model_dump(mode="json") for g in gaps],
            "query_budget": {"remaining_queries": remaining, "remaining_rounds": rounds, "max_queries_this_round": per_round},
            "instructions": "Plan focused queries within this round's allowance; reserve remaining queries for later grader-directed gaps. Do not repeat prior queries.",
        }, query_schema(per_round), light=True)
        def normalize(s: str) -> str:
            return " ".join(s.casefold().split())
        normalized = [normalize(q) for q in plan.queries]
        if len(set(normalized)) != len(normalized) or set(normalized) & {normalize(q) for q in prior}:
            raise ValueError("repeated query makes no progress")
        if len(prior) + len(plan.queries) > task.budget.max_queries:
            raise ValueError("query plan exceeds remaining task budget")
        batches = []
        business = getattr(execution, "business_context", None)
        if business is not None:
            business = QueryCompiler.for_task(business, task.objective, task_ref=f"task:{task.key}:{task_revision}", document_ids=task.sources.document_ids)
        for query in plan.queries:
            await execution.debit("queries")
            # Selectors come from the validated task. There is no model argument
            # that can broaden this hard scope or supply a different tenant.
            batch = await self.retrieval.retrieve_candidates(
                RetrievalRequest(query=query, document_ids=task.sources.document_ids), execution.scope, execution.snapshot,
                **business_kwargs(business),
            )
            if any(hit.user_id != execution.scope.user_id or (
                task.sources.document_ids and hit.document_id not in task.sources.document_ids
            ) for hit in batch.hits):
                raise ValueError("retrieval violated task source scope")
            batches.append(batch)
        hits = {hit.child_id: hit for batch in batches for hit in batch.hits}
        summary = await execution.structured("retrieval_summary", {
            "task": task.model_dump(mode="json"), "candidates": [h.model_dump(mode="json") for h in hits.values()],
            "instructions": "Return candidate findings with child_id references. No arithmetic or factual acceptance.",
        }, CandidateSummary, light=True)
        criteria = {c.id for c in task.criteria}
        validate_findings(summary.findings, set(hits), criteria)
        if len(summary.findings) > task.output.max_items or any(g.criterion_id not in criteria for g in summary.gaps):
            raise ValueError("summary exceeds task output scope")
        return CandidateResult(
            origin_key=task.key, task_revision=task_revision, input_version=input_version, round_id=round_id,
            criterion_ids=tuple(c.id for c in task.criteria), candidates=tuple(batches),
            findings=summary.findings, gaps=summary.gaps, queries=plan.queries,
            context={"queries": [*prior, *plan.queries], "summary": [f.model_dump(mode="json") for f in summary.findings]},
        )


class GroupGrader:
    async def grade(self, bundle: PreparedBundle, execution: Execution, *, source_contracts: Sequence[dict[str, Any]] = ()) -> GradeResult:
        grade = await execution.structured("evidence_grader", {
            "bundle_hash": bundle.bundle_hash, "criteria": [c.model_dump(mode="json") for c in bundle.criteria],
            "evidence": bundle.packed.rendered_context, "evidence_ids": list(bundle.packed.manifest),
            "evidence_requirements": [r.model_dump(mode="json") for r in bundle.packed.evidence_requirements],
            "scope_evidence_missing": bundle.packed.scope_evidence_missing,
            "evidence_manifest": {k: v.model_dump(mode="json") for k, v in bundle.packed.manifest.items()},
            "source_contracts": list(source_contracts),
            "instructions": "Assess each criterion, source/entity/period/unit/scope and contradictions. Unsupported facts are gaps. Resolved scope requirements mean source availability only; semantic_status pending requires your applicability judgment, including exclusions. A mere company mention/customer example is not applicability evidence. Query/profile text is not evidence. Do not calculate.",
        }, GradeResult, light=True)
        validate_grade(grade, bundle)
        return grade


class EvidenceInput(Contract):
    result_ref: Key
    bundle: PreparedBundle
    grade: GradeResult


class Operand(Contract):
    source_kind: Literal["evidence", "user_input"] = Field(default="evidence", description="Explicitly user_input when evidence_ref=query; otherwise evidence.")
    value: float = Field(allow_inf_nan=False)
    evidence_ref: Key
    bundle_hash: str
    quote: Text = Field(description="Exact contiguous substring of the original source, copied verbatim. Never paraphrase, shorten internally or invent a source quote.")
    unit: Text = Field(description="Unit copied from the source; user-input arithmetic may use number.")
    entity: Text = Field(description="Exact entity name copied from the source, without appending the metric or year. User-input arithmetic may use user_input.")
    period: Text = Field(description="Period copied from the source; user-input arithmetic may use unspecified.")

    @model_validator(mode="after")
    def query_binding(self) -> "Operand":
        if (self.evidence_ref == "query") != (self.source_kind == "user_input"):
            raise ValueError("query operands require source_kind=user_input; external operands require source_kind=evidence")
        return self


class CalculationRequest(Contract):
    id: Key
    operation: Literal["sum", "difference", "ratio", "growth_percent", "mean"]
    operands: tuple[Operand, ...] = Field(min_length=1, max_length=16)
    criterion_ids: tuple[Key, ...] = Field(min_length=1, max_length=32)


class VerifiedCalculation(CalculationRequest):
    expression: str
    value: float = Field(allow_inf_nan=False)
    unit: str


class AnalysisDraft(CandidateSummary):
    calculations: tuple[CalculationRequest, ...] = Field(default=(), max_length=16)


def analysis_schema(max_items: int, *, question: str = "", evidence_refs: set[str] | None = None,
                    inputs: Sequence[EvidenceInput] | None = None, allow_calculations: bool = True) -> type[AnalysisDraft]:
    """Check output quota and user-input bindings inside the bounded repair path."""
    class BoundedAnalysisDraft(AnalysisDraft):
        @model_validator(mode="after")
        def output_budget(self) -> "BoundedAnalysisDraft":
            errors = []
            if self.calculations and not allow_calculations:
                errors.append("task does not authorize calculation")
            if len(self.findings) + len(self.calculations) > max_items:
                errors.append(f"findings plus calculations must total at most {max_items}; {len(self.calculations)} calculations already consume that many slots. Omit explanatory findings, using findings=[] when calculations fill the quota")
            if question:
                for calculation in self.calculations:
                    for operand in calculation.operands:
                        if operand.source_kind == "user_input":
                            try:
                                validate_user_operand(operand, question)
                            except ValueError as error:
                                errors.append(str(error))
            if evidence_refs is not None:
                allowed = set(evidence_refs)
                if question and any(o.source_kind == "user_input" for c in self.calculations for o in c.operands):
                    allowed.add("query")
                if any(not set(f.evidence_refs) <= allowed for f in self.findings):
                    errors.append("finding source is not in the permitted evidence; cite accepted evidence IDs for document-derived findings. query is only for user-input arithmetic, not calculations based on documents")
            if inputs is not None:
                verified = []
                for request in self.calculations if allow_calculations else ():
                    try:
                        verified.append(calculate(request, inputs, question=question))
                    except ValueError:
                        # Missing operands still become explicit gaps in the
                        # executor; they never authorize extra numeric claims.
                        pass
                unknown = analysis_unknown_numbers(self.findings, verified, inputs, question)
                if unknown:
                    errors.append(f"unsupported numeric claims {unknown}: omit derived numbers absent from sources and verified calculations, including uncomputed percentage-point differences")
            if errors:
                raise ValueError("; ".join(dict.fromkeys(errors)))
            return self
    return BoundedAnalysisDraft


class AnalysisResult(Contract):
    task_key: Key
    task_revision: int = 1
    input_version: int = 1
    round_id: int = 1
    dependency_results: dict[str, str]
    status: Literal["completed", "needs_input"]
    findings: tuple[Finding, ...]
    calculations: tuple[VerifiedCalculation, ...]
    gaps: tuple[Gap, ...]
    context: dict[str, Any]


def validate_user_operand(operand: Operand, question: str) -> None:
    errors = []
    if not question or operand.evidence_ref != "query" or operand.bundle_hash != content_hash({"question": question}) or operand.quote not in question:
        errors.append("operand is not bound to the original user input")
    for name, value, generic in (
        ("unit", operand.unit, "number"), ("entity", operand.entity, "user_input"), ("period", operand.period, "unspecified"),
    ):
        if value != generic and value not in question:
            errors.append(f"operand metadata {name}={value!r} is absent from the original user input; copy its exact label or use {name}={generic!r}")
    if errors:
        raise ValueError("; ".join(errors))


def calculate(request: CalculationRequest, inputs: Sequence[EvidenceInput], *, question: str = "") -> VerifiedCalculation:
    indexed = {i.bundle.bundle_hash: i for i in inputs}
    for operand in request.operands:
        if operand.source_kind == "user_input":
            validate_user_operand(operand, question)
        else:
            value = indexed.get(operand.bundle_hash)
            if value is None:
                raise ValueError("operand refers to an unaccepted bundle")
            allowed = {r for c in value.grade.coverage if c.status == "supported" for r in c.evidence_refs}
            item = next((e for e in value.bundle.packed.items if e.evidence_id == operand.evidence_ref and e.evidence_id in allowed), None)
            if item is None or operand.quote not in item.content:
                raise ValueError("operand quote is absent from accepted evidence")
            source_context = " ".join((*item.heading_path, operand.quote))
            if any(v not in source_context for v in (operand.unit, operand.entity, operand.period)):
                raise ValueError("operand entity/period/unit lacks source context")
        numbers = {float(n.replace(",", "")) for n in re.findall(r"(?<![0-9A-Za-z_.])-?\d[\d,]*(?:\.\d+)?(?![0-9A-Za-z_])", operand.quote)}
        if operand.value not in numbers:
            raise ValueError("operand value is absent from its source quote")
    operands = request.operands
    if len({o.unit for o in operands}) != 1:
        raise ValueError("unit conversion needs explicit sourced normalization")
    if request.operation in ("difference", "ratio", "growth_percent") and len(operands) != 2:
        raise ValueError("binary operation requires two operands")
    if request.operation == "growth_percent" and any(o.source_kind == "evidence" for o in operands) and (len({o.entity for o in operands}) != 1 or operands[0].period == operands[1].period):
        raise ValueError("growth requires the same entity in distinct periods")
    nums = [repr(o.value) for o in operands]
    expression = {
        "sum": lambda: "+".join(nums), "mean": lambda: f"({'+'.join(nums)})/{len(nums)}",
        "difference": lambda: f"{nums[1]}-{nums[0]}", "ratio": lambda: f"{nums[1]}/{nums[0]}",
        "growth_percent": lambda: f"({nums[1]}-{nums[0]})/{nums[0]}*100",
    }[request.operation]()
    result = SafeCalculator().evaluate(expression)
    return VerifiedCalculation(**request.model_dump(), expression=expression, value=result,
                               unit="%" if request.operation == "growth_percent" else "ratio" if request.operation == "ratio" else operands[0].unit)


class AnalysisAgent:
    async def execute(
        self, task: TaskSpec, inputs: Sequence[EvidenceInput], context: dict[str, Any], execution: Execution, *,
        dependency_results: dict[str, str], task_revision: int = 1, input_version: int = 1, round_id: int = 1,
    ) -> AnalysisResult:
        if task.agent_type != "analysis" or (inputs and "read_evidence" not in task.tools.allowed):
            raise ValueError("analysis role requires read_evidence")
        question = getattr(execution, "question", "")
        if (not inputs and not question) or {i.result_ref for i in inputs} != set(dependency_results.values()):
            raise ValueError("analysis must pin every dependency result")
        for value in inputs:
            validate_grade(value.grade, value.bundle)
            if task.sources.document_ids and not {e.document_id for e in value.bundle.packed.items} <= set(task.sources.document_ids):
                raise ValueError("analysis evidence exceeds source scope")
        allowed = {r for i in inputs for c in i.grade.coverage if c.status == "supported" for r in c.evidence_refs}
        draft = await execution.structured("analysis", {
            "task": task.model_dump(mode="json"), "context": context,
            "inputs": [v.model_dump(mode="json") for v in inputs],
            "user_input": {"question": question, "input_hash": content_hash({"question": question}), "evidence_ref": "query"},
            "instructions": "Cite accepted evidence; declare calculations using quoted operands and operations. For user-provided numbers use source_kind=user_input, the original question hash and evidence_ref=query. Use generic metadata unit=number, entity=user_input, period=unspecified, or labels appearing verbatim in the original question. For user-only arithmetic, findings can be empty; explanatory fact/inference findings may reference query only alongside valid user-input calculations. Findings may otherwise cite only accepted evidence IDs and must not duplicate calculated outcomes. The executor creates calculation findings. List operands in baseline-then-current order for growth_percent, difference and ratio. Report missing facts as gaps, never invent operands. Arithmetic values are computed by the executor.",
        }, analysis_schema(task.output.max_items, question=question, evidence_refs=allowed, inputs=inputs,
                           allow_calculations="calculate" in task.tools.allowed))
        if draft.calculations and "calculate" not in task.tools.allowed:
            raise ValueError("task does not authorize calculation")
        criteria = {c.id for c in task.criteria}
        verified = []
        gaps = list(draft.gaps)
        if any(g.criterion_id not in criteria for g in gaps):
            raise ValueError("analysis gap is outside task scope")
        if len({c.id for c in draft.calculations}) != len(draft.calculations):
            raise ValueError("duplicate calculation key")
        for request in draft.calculations:
            if not set(request.criterion_ids) <= criteria:
                raise ValueError("calculation exceeds task scope")
            try:
                verified.append(calculate(request, inputs, question=question))
            except ValueError as error:
                gaps.append(Gap(criterion_id=request.criterion_ids[0], description=str(error), search_direction="Retrieve the missing comparable source operands"))
        # Match durable submission: query provenance becomes available only
        # after user-input operands have actually passed source validation.
        if any(o.source_kind == "user_input" for calculation in verified for o in calculation.operands):
            allowed.add("query")
        if any(f.kind == "calculation" for f in draft.findings):
            raise ValueError("calculation findings are generated by the executor")
        if analysis_unknown_numbers(draft.findings, verified, inputs, question):
            raise ValueError("unsupported numeric claims in analysis findings")
        validate_findings(draft.findings, allowed, criteria, calculations={c.id for c in verified})
        findings = [*draft.findings, *(
            Finding(statement=f"{c.id}: {c.expression} = {c.value:g} {c.unit}", kind="calculation",
                    evidence_refs=tuple(dict.fromkeys(o.evidence_ref for o in c.operands)),
                    calculation_refs=(c.id,), criterion_ids=c.criterion_ids)
            for c in verified
        )]
        if len(findings) > task.output.max_items:
            raise ValueError("analysis output exceeds contract")
        if not findings and not gaps:
            gaps.append(Gap(criterion_id=task.criteria[0].id, description="No supported analysis findings"))
        return AnalysisResult(
            task_key=task.key, task_revision=task_revision, input_version=input_version, round_id=round_id,
            dependency_results=dependency_results, status="needs_input" if gaps else "completed",
            findings=tuple(findings), calculations=tuple(verified), gaps=tuple(gaps),
            context={"summary": [f.model_dump(mode="json") for f in findings]},
        )


def analysis_unknown_numbers(findings: Sequence[Finding], calculations: Sequence[VerifiedCalculation],
                             inputs: Sequence[EvidenceInput], question: str) -> list[str]:
    return unsupported_numbers("\n".join(f.statement for f in findings),
        [question, *(evidence_text(item.model_dump(mode="json")) for value in inputs for item in value.bundle.packed.items)],
        [calculation.model_dump(mode="json") for calculation in calculations])
