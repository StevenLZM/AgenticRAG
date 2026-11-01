"""Persist prepared/graded versions and enforce deterministic acceptance gates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.contracts import Gap, GradeResult, PreparedBundle, content_hash
from agentic_rag.orchestration.store_sql import enqueue, lock_run, save_run

if TYPE_CHECKING:
    from agentic_rag.orchestration.store import CoordinationStore


def validate_grade(grade: GradeResult, bundle: PreparedBundle) -> None:
    if grade.decision == "sufficient" and bundle.packed.scope_evidence_missing:
        raise ValueError("scope_evidence_missing prevents sufficient grade")
    if grade.bundle_hash != bundle.bundle_hash:
        raise ValueError("grade binds a different evidence bundle")
    criteria = {c.id for c in bundle.criteria}
    coverage = {c.criterion_id for c in grade.coverage}
    if coverage != criteria or len(grade.coverage) != len(criteria):
        raise ValueError("grade must cover each criterion exactly once")
    allowed = set(bundle.packed.manifest)
    for item in grade.coverage:
        if not set(item.evidence_refs) <= allowed or (item.status == "supported" and not item.evidence_refs):
            raise ValueError("grade has unsupported evidence references")
    for finding in grade.findings:
        if not finding.evidence_refs or not set(finding.evidence_refs) <= allowed:
            raise ValueError("finding has no verified source")
    for gap in grade.gaps:
        if gap.criterion_id not in criteria or not set(gap.evidence_refs) <= allowed:
            raise ValueError("gap has invalid criterion or evidence")
    supported = {c.criterion_id for c in grade.coverage if c.status == "supported"}
    if grade.decision == "sufficient" and not {c.id for c in bundle.criteria if c.required} <= supported:
        raise ValueError("sufficient grade lacks required coverage")


async def constrain_sources(db: AsyncSession, run: dict[str, Any], item: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """A model's sufficiency cannot waive a frozen must-use source contract."""
    group = run["data"]["groups"][item["payload"]["group_key"]]
    raw = (await db.execute(select(schema.results.c.payload).where(schema.results.c.id == item["payload"]["bundle_ref"]))).scalar_one()
    bundle, grade = PreparedBundle.model_validate(raw), GradeResult.model_validate(payload)
    validate_grade(grade, bundle)
    blocked: set[str] = set()
    gaps: list[Gap] = []
    for member in group["spec"]["members"]:
        spec = run["data"]["tasks"][member["task_key"]]["spec"]
        criteria = {c["id"] for c in spec["criteria"]}
        refs = {r for c in grade.coverage if c.criterion_id in criteria and c.status == "supported" for r in c.evidence_refs}
        used = {bundle.packed.manifest[ref].document_id for ref in refs}
        missing = set(spec["sources"]["required_document_ids"]) - used
        if missing:
            blocked.update(criteria)
            gaps.extend(Gap(criterion_id=key, description="Required source not supported: " + ", ".join(sorted(missing)),
                            related_task_keys=(member["task_key"],)) for key in sorted(criteria))
    group["contract_gaps"] = [g.model_dump(mode="json") for g in gaps]
    if not gaps:
        return payload
    return grade.model_copy(update={
        "decision": "insufficient" if grade.decision == "sufficient" else grade.decision,
        "coverage": tuple(c.model_copy(update={"status": "unsupported", "evidence_refs": (), "explanation": "Required source contract not satisfied"}) if c.criterion_id in blocked else c for c in grade.coverage),
        "gaps": tuple((*grade.gaps, *gaps))[:32],
    }).model_dump(mode="json")


async def complete_evidence_stage(
    db: AsyncSession, run: dict[str, Any], item: dict[str, Any], result_id: str, payload: dict[str, Any],
) -> None:
    group_key = item["payload"]["group_key"]
    group = run["data"]["groups"][group_key]
    frozen = (await db.execute(select(schema.snapshots.c.data).where(schema.snapshots.c.id == group["snapshot_id"]))).scalar_one()
    if item["stage"] == "prepare":
        bundle = PreparedBundle.model_validate(payload)
        candidates = []
        for member in frozen["members"]:
            for ref in member["results"]:
                row = (await db.execute(select(schema.results).where(schema.results.c.id == ref["result_id"]))).mappings().one()
                if row["content_hash"] != ref["hash"]:
                    raise ValueError("immutable candidate hash mismatch")
                candidates.append(row["payload"])
        if bundle.input_hash != content_hash(candidates) or [c.model_dump(mode="json") for c in bundle.criteria] != frozen["criteria"]:
            raise ValueError("prepared evidence does not bind the frozen input")
        group["prepared_ref"], group["status"] = result_id, "grading"
        await enqueue(db, run["id"], "grade",
                      f"grade:{group_key}:{group['group_revision']}:{group['aggregation_version']}",
                      {**item["payload"], "bundle_ref": result_id})
    else:
        if group.get("prepared_ref") != item["payload"].get("bundle_ref"):
            raise ValueError("grade input was replaced")
        prepared = (await db.execute(select(schema.results.c.payload).where(schema.results.c.id == group["prepared_ref"]))).scalar_one()
        validate_grade(GradeResult.model_validate(payload), PreparedBundle.model_validate(prepared))
        group["grade_ref"], group["status"] = result_id, "feedback_ready"
        for member in frozen["members"]:
            if member.get("origin") == "fast_pipeline":
                continue
            task = run["data"]["tasks"][member["task_key"]]
            if task["state"] == "waiting_grade":
                task["state"] = "waiting_feedback"


async def accept_grade(store: CoordinationStore, run_id: str, group_key: str, result_id: str) -> None:
    async with store.session_factory.begin() as db:
        run = await lock_run(db, run_id)
        await accept_locked(db, run, group_key, result_id)
        await save_run(db, run)


async def accept_locked(db: AsyncSession, run: dict[str, Any], group_key: str, result_id: str) -> None:
    from agentic_rag.orchestration.store import StaleClaim
    group = run["data"]["groups"][group_key]
    if run["status"] != "active" or run["data"].get("answer") is not None or group.get("grade_ref") != result_id:
        raise StaleClaim("grade is no longer the current group feedback")
    if group.get("accepted_grade_ref") == result_id:
        return
    payload = (await db.execute(select(schema.results.c.payload).where(schema.results.c.id == result_id))).scalar_one()
    grade = GradeResult.model_validate(payload)
    prepared = (await db.execute(select(schema.results.c.payload).where(schema.results.c.id == group["prepared_ref"]))).scalar_one()
    bundle = PreparedBundle.model_validate(prepared)
    validate_grade(grade, bundle)
    frozen = (await db.execute(select(schema.snapshots.c.data).where(schema.snapshots.c.id == group["snapshot_id"]))).scalar_one()
    supported = {c.criterion_id for c in grade.coverage if c.status == "supported"}
    complete = frozen["collection_completeness"] == "complete" and grade.decision == "sufficient"
    group["acceptance"] = "accepted" if complete else "partial" if supported else "rejected"
    group["accepted_grade_ref"] = result_id
    for member in frozen["members"]:
        if member.get("origin") == "fast_pipeline":
            continue
        task = run["data"]["tasks"][member["task_key"]]
        required = {c["id"] for c in task["spec"]["criteria"] if c["required"]}
        supplied = supported & {c["id"] for c in task["spec"]["criteria"]}
        accepted = bool(required <= supplied) and member["status"] == "succeeded"
        evidence = {content_hash({k: v for k, v in item.model_dump(mode="json").items() if k not in ("evidence_id", "covered_target_ids")})
                    for item in bundle.packed.items if member["task_key"] in bundle.source_map.get(item.evidence_id, ())}
        previous = task.get("progress")
        gained = previous is None or bool(evidence - set(previous["evidence"])) or bool(supplied - set(previous["supported"]))
        task["no_progress_rounds"] = 0 if gained else task.get("no_progress_rounds", 0) + 1
        task["progress"] = {"evidence": sorted(evidence | set(previous["evidence"] if previous else ())),
                            "supported": sorted(supplied | set(previous["supported"] if previous else ()))}
        task["supported_criteria"] = sorted(supplied)
        task["acceptance"] = "accepted" if accepted else "partial" if supplied else "rejected"
        task["accepted_ref"] = result_id if supplied else None
        if accepted:
            task["state"] = "completed"
        elif task["no_progress_rounds"] >= task["spec"]["budget"]["max_no_progress_rounds"]:
            task["state"] = "blocked"
    run["data"]["accepted"][group_key] = {
        "grade_ref": result_id, "bundle_ref": group["prepared_ref"], "snapshot_id": group["snapshot_id"],
        "acceptance": group["acceptance"], "supported_criteria": sorted(supported),
    }
