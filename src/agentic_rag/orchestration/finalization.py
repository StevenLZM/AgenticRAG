"""Freeze accepted immutable inputs before any public model token is emitted."""
from typing import Any
import json

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.orchestration import schema
from agentic_rag.orchestration.contracts import GradeResult, PreparedBundle, content_hash
from agentic_rag.orchestration.store_sql import enqueue


def bound_inputs(evidence: list[dict[str, Any]], analyses: list[dict[str, Any]], assessments: list[dict[str, Any]],
                 groups: dict[str, list[str]], *, budget: int, mode: str):
    """Keep complete supporting units; never trim a quote under an old grade."""
    def cost(items, reports):
        return len(json.dumps({"evidence": items, "analyses": reports}, ensure_ascii=False))
    if cost(evidence, analyses) <= budget:
        return evidence, analyses, assessments, []
    if mode == "full":
        raise ValueError("accepted inputs exceed the answer context budget; select a partial answer or narrow the task")
    selected: dict[str, dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    indexed = {item["evidence_id"]: item for item in evidence}
    # A verified computation is indivisible: its whole source excerpts travel
    # with it, or the computation is explicitly omitted.
    for report in analyses:
        result = report["result"]
        refs = {r for f in result["findings"] for r in f["evidence_refs"] if r != "query"}
        refs.update(o["evidence_ref"] for c in result["calculations"] for o in c["operands"] if o["source_kind"] == "evidence")
        combined = {**selected, **{key: indexed[key] for key in refs if key in indexed}}
        if refs <= indexed.keys() and cost(list(combined.values()), [*reports, report]) <= budget:
            selected, reports = combined, [*reports, report]
        else:
            gaps.append({"missing_analysis": report["task_key"], "reason": "answer_context_budget"})
    for key, group_refs in groups.items():
        combined = {**selected, **{ref: indexed[ref] for ref in group_refs}}
        if cost(list(combined.values()), reports) <= budget:
            selected = combined
        else:
            gaps.append({"partial_group": key, "reason": "answer_context_budget"})
    assessments = [{**a, "acceptance": a["acceptance"] if set(groups[a["group_key"]]) <= selected.keys() else "partial"} for a in assessments]
    return list(selected.values()), reports, assessments, gaps


async def freeze_answer(db: AsyncSession, run: dict[str, Any], *, mode: str = "full", clarification: str = "") -> None:
    data = run["data"]
    if data.get("answer") is not None:
        return
    if mode == "full":
        has_analysis = any(t["spec"]["agent_type"] == "analysis" and t["acceptance"] == "accepted" for t in data["tasks"].values())
        if (not data["accepted"] and not has_analysis) or any(t["spec"]["required_for_answer"] and t["acceptance"] != "accepted" for t in data["tasks"].values()):
            raise ValueError("full answer requires all required accepted tasks")
        if any(g.get("acceptance") != "accepted" for g in data["groups"].values() if g.get("origin") != "fast_pipeline" or data["route"] == "fast"):
            raise ValueError("full answer cannot hide partial evidence collection")
        if data.get("requires_analysis") and not any(t["spec"]["agent_type"] == "analysis" and t["acceptance"] == "accepted" for t in data["tasks"].values()):
            raise ValueError("analytical query requires accepted analysis")
    evidence: dict[str, dict[str, Any]] = {}
    analyses: list[dict[str, Any]] = []
    assessments: list[dict[str, Any]] = []
    limitations: list[dict[str, Any]] = []
    group_evidence: dict[str, list[str]] = {}
    for key, accepted in data["accepted"].items():
        grade_raw = (await db.execute(select(schema.results.c.payload).where(schema.results.c.id == accepted["grade_ref"]))).scalar_one()
        grade = GradeResult.model_validate(grade_raw)
        bundle_raw = (await db.execute(select(schema.results.c.payload).where(schema.results.c.id == accepted["bundle_ref"]))).scalar_one()
        bundle = PreparedBundle.model_validate(bundle_raw)
        allowed = {ref for c in grade.coverage if c.status == "supported" for ref in c.evidence_refs}
        group_evidence[key] = sorted(allowed)
        for item in bundle.packed.items:
            if item.evidence_id in allowed:
                previous = evidence.get(item.evidence_id)
                raw = item.model_dump(mode="json")
                if previous is not None and {k: v for k, v in previous.items() if k not in ("covered_target_ids", "content")} != {k: v for k, v in raw.items() if k not in ("covered_target_ids", "content")}:
                    # IDs are local to a pack. Preserve an unambiguous global
                    # locator rather than silently conflating two evidence items.
                    raise ValueError("conflicting evidence IDs across accepted bundles")
                if previous is not None:
                    raw["covered_target_ids"] = sorted(set(previous["covered_target_ids"]) | set(raw["covered_target_ids"]))
                    if raw["content"] in previous["content"]:
                        raw["content"] = previous["content"]
                    elif previous["content"] not in raw["content"]:
                        # Different excerpts of the same immutable source both
                        # retain attribution; neither may overwrite the other.
                        raw["content"] = previous["content"] + "\n" + raw["content"]
                evidence[item.evidence_id] = raw
        assessments.append({"group_key": key, "grade_ref": accepted["grade_ref"], "bundle_hash": bundle.bundle_hash, "acceptance": accepted["acceptance"]})
        limitations.extend(g.model_dump(mode="json") for g in grade.gaps)
        frozen = (await db.execute(select(schema.snapshots.c.data).where(schema.snapshots.c.id == accepted["snapshot_id"]))).scalar_one()
        limitations.extend({"missing_task": k} for k in frozen["missing_members"])
    for key, task in data["tasks"].items():
        if task["spec"]["agent_type"] == "analysis" and task["acceptance"] == "accepted":
            raw = (await db.execute(select(schema.results.c.payload).where(schema.results.c.id == task["analysis_result_ref"]))).scalar_one()
            report = {k: v for k, v in raw.items() if k in ("status", "findings", "calculations", "gaps")}
            analyses.append({"task_key": key, "result_ref": task["analysis_result_ref"], "assessment_ref": task["accepted_ref"], "result": report})
        elif task["spec"]["required_for_answer"] and task["acceptance"] != "accepted":
            limitations.append({"missing_task": key, "state": task["state"]})
    selected, analyses, assessments, omissions = bound_inputs(list(evidence.values()), analyses, assessments, group_evidence,
        # Preserve the historical limit for existing Runs; initialization must
        # never backfill or reset a persisted run's execution boundaries.
        budget=data.get("answer_input_char_limit", data["snapshot"]["max_evidence_tokens"]), mode=mode)
    limitations.extend(omissions)
    frozen_input = {"question": data["question"], "mode": "clarification" if clarification else mode,
                    "clarification": clarification, "evidence": selected, "analyses": analyses,
                    "assessments": assessments, "limitations": limitations}
    from agentic_rag.orchestration.authorization import authorize_sources
    await authorize_sources(db, run, frozen_input["evidence"])
    data["answer"] = {"input": frozen_input, "input_hash": content_hash(frozen_input), "epoch": 1,
                      "seq": 0, "text": "", "status": "frozen", "faithfulness": "pending"}
    data["phase"] = "answering"
    await db.execute(update(schema.work_items).where(
        schema.work_items.c.run_id == run["id"], schema.work_items.c.status.in_(("queued", "running", "retry_wait")),
    ).values(status="cancelled", lease_until=None))
    await enqueue(db, run["id"], "answer", "answer:1", {"epoch": 1, "input_hash": data["answer"]["input_hash"]})
