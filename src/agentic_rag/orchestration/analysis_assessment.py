"""Separate analytical acceptance, tied to exact evidence dependency versions."""
from typing import Any, Literal

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.agents import AnalysisResult, CalculationRequest, EvidenceInput, Execution, calculate, validate_findings
from agentic_rag.orchestration.contracts import Contract, Gap, Key, TaskSpec, content_hash
from agentic_rag.orchestration.store_sql import enqueue


class AnalysisAssessment(Contract):
    result_hash: str
    decision: Literal["accepted", "needs_input"]
    supported_criteria: tuple[Key, ...] = Field(default=(), max_length=32)
    gaps: tuple[Gap, ...] = Field(default=(), max_length=32)


async def load_inputs(db: AsyncSession, references: dict[str, str]) -> list[EvidenceInput]:
    values = []
    for result_ref in dict.fromkeys(references.values()):
        row = (await db.execute(select(schema.results).where(schema.results.c.id == result_ref))).mappings().one()
        if row["stage"] != "grade":
            raise ValueError("analysis dependencies must reference graded retrieval inputs")
        work = (await db.execute(select(schema.work_items).where(schema.work_items.c.id == row["work_id"]))).mappings().one()
        bundle = (await db.execute(select(schema.results.c.payload).where(schema.results.c.id == work["payload"]["bundle_ref"]))).scalar_one()
        values.append(EvidenceInput.model_validate({"result_ref": result_ref, "bundle": bundle, "grade": row["payload"]}))
    return values


async def assess(task: TaskSpec, result: AnalysisResult, inputs: list[EvidenceInput], execution: Execution) -> AnalysisAssessment:
    assessment = await execution.structured("analysis_assessment", {
        "task": task.model_dump(mode="json"), "result": result.model_dump(mode="json"), "result_hash": content_hash(result),
        "evidence": [i.model_dump(mode="json") for i in inputs],
        "user_question": getattr(execution, "question", ""),
        "instructions": "Verify conclusions, operand attribution, period ordering, units, scope and every required criterion. Calculator success alone is not factual acceptance. Assess semantic support across structured calculations, verbatim operands, accepted evidence and the original user question. A verified expression satisfies an equivalent formula requirement; final prose/table formatting is performed later by the answer stage. Do not reject solely because that prose is not yet rendered. Return needs_input with directed gaps for unsupported facts, incomparable operands or reasoning.",
    }, AnalysisAssessment, light=True)
    validate_assessment(task, result, assessment)
    return assessment


def validate_assessment(task: TaskSpec, result: AnalysisResult, assessment: AnalysisAssessment) -> None:
    if assessment.result_hash != content_hash(result):
        raise ValueError("analysis assessment input mismatch")
    allowed = {c.id for c in task.criteria}
    if not set(assessment.supported_criteria) <= allowed or any(g.criterion_id not in allowed for g in assessment.gaps):
        raise ValueError("analysis assessment exceeds criterion scope")
    if assessment.decision == "accepted" and (result.status != "completed" or assessment.gaps or not {c.id for c in task.criteria if c.required} <= set(assessment.supported_criteria)):
        raise ValueError("analysis acceptance lacks complete supported criteria")


async def complete_analysis(db: AsyncSession, run: dict[str, Any], item: dict[str, Any], result_id: str, payload: dict[str, Any]) -> None:
    task = run["data"]["tasks"][item["payload"]["task_key"]]
    spec = TaskSpec.model_validate(task["spec"])
    if item["stage"] == "analysis":
        result = AnalysisResult.model_validate(payload)
        if (result.task_key != spec.key or result.task_revision != task["revision"] or result.input_version != task["input_version"]
                or result.round_id != task["round_id"] or result.dependency_results != item["payload"]["dependency_results"]):
            raise ValueError("analysis identity or dependency mismatch")
        inputs = await load_inputs(db, result.dependency_results)
        allowed = {ref for i in inputs for c in i.grade.coverage if c.status == "supported" for ref in c.evidence_refs}
        if any(o.source_kind == "user_input" for c in result.calculations for o in c.operands):
            allowed.add("query")
        validate_findings(result.findings, allowed, {c.id for c in spec.criteria}, calculations={c.id for c in result.calculations})
        for calculation in result.calculations:
            checked = calculate(CalculationRequest.model_validate(calculation.model_dump(exclude={"expression", "value", "unit"})), inputs, question=run["data"]["question"])
            if checked != calculation:
                raise ValueError("analysis calculation differs from recomputation")
        task["analysis_result_ref"], task["context_ref"], task["state"] = result_id, result_id, "waiting_assessment"
        task["dependency_results"] = result.dependency_results
        await enqueue(db, run["id"], "analysis_grade", f"analysis-grade:{spec.key}:{task['revision']}:{task['input_version']}:{task['round_id']}",
                      {**item["payload"], "analysis_ref": result_id})
    else:
        result_raw = (await db.execute(select(schema.results.c.payload).where(schema.results.c.id == item["payload"]["analysis_ref"]))).scalar_one()
        result = AnalysisResult.model_validate(result_raw)
        validate_assessment(spec, result, AnalysisAssessment.model_validate(payload))
        task["assessment_ref"], task["state"] = result_id, "waiting_feedback"


async def accept_analysis(db: AsyncSession, run: dict[str, Any], result_id: str, payload: dict[str, Any]) -> None:
    task = next((t for t in run["data"]["tasks"].values() if t.get("assessment_ref") == result_id), None)
    if task is None:
        return
    row = (await db.execute(select(schema.results).where(schema.results.c.id == result_id))).mappings().one()
    work = (await db.execute(select(schema.work_items).where(schema.work_items.c.id == row["work_id"]))).mappings().one()
    binding = work["payload"]
    current_inputs = {d["task_key"]: run["data"]["tasks"][d["task_key"]].get("accepted_ref") for d in task["spec"]["dependencies"]}
    if (row["stage"] != "analysis_grade" or any(binding[name] != task[name] for name in ("revision", "input_version", "round_id"))
            or binding["dependency_results"] != current_inputs or binding["analysis_ref"] != task.get("analysis_result_ref")):
        return  # Durable receipt is not acceptance of a superseded input.
    assessment = AnalysisAssessment.model_validate(payload)
    raw = (await db.execute(select(schema.results.c.payload).where(schema.results.c.id == binding["analysis_ref"]))).scalar_one()
    result = AnalysisResult.model_validate(raw)
    if result.dependency_results != current_inputs:
        return
    validate_assessment(TaskSpec.model_validate(task["spec"]), result, assessment)
    task["acceptance"] = "accepted" if assessment.decision == "accepted" else "rejected"
    task["accepted_ref"] = result_id if assessment.decision == "accepted" else None
    task["state"] = "completed" if assessment.decision == "accepted" else "needs_input"
    task["supported_criteria"] = list(assessment.supported_criteria)
    run["data"].setdefault("feedback", []).append({"task_key": task["spec"]["key"], "assessment": payload,
        "findings": raw["findings"], "gaps": [*raw["gaps"], *payload["gaps"]]})
