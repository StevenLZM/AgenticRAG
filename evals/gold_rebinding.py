"""Publish immutable v4 bindings without altering questions or scoring semantics."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.corpus_snapshot import verify_originals
from evals.formal_gold import bind_anchor
from evals.gold_answer_spec import GoldAnswerSpec, load_gold_answer_specs
from evals.gold_v2_models import CorpusSnapshot, GoldCaseV2, canonical_hash
from evals.gold_v2_validation import load_gold_v2, strict_json
from evals.migration_evidence import verify_migration_equivalence
from evals.report import atomic_write_text

BUNDLE_FILES = ("snapshot.json", "gold.v2.2.jsonl", "answer-spec.jsonl", "migration-evidence.json", "rebind-audit.json")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path):
    return strict_json(path.read_text(encoding="utf-8"))


def _rows(path):
    return [strict_json(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _write(path, value, *, rows=False):
    text = ("".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in value) if rows else
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    atomic_write_text(path, text)
    path.chmod(0o600)


def rebind_gold_rows(rows: list[dict[str, Any]], *, snapshot_id: str) -> list[dict[str, Any]]:
    return [{**deepcopy(row), "snapshot_id": snapshot_id} for row in rows]


def rebind_spec_rows(rows: list[dict[str, Any]], *, cases: list[GoldCaseV2]) -> list[dict[str, Any]]:
    by_id = {c.case_id: c for c in cases}
    if len(rows) != len(by_id) or {r["case_id"] for r in rows} != set(by_id):
        raise ValueError("sidecar membership changed")
    return [{**deepcopy(row), "snapshot_id": by_id[row["case_id"]].snapshot_id,
             "gold_case_sha256": canonical_hash(by_id[row["case_id"]].model_dump(mode="json"))} for row in rows]


def validate_sufficient_sets(cases, snapshot):
    """Reprove each annotated set; additional valid alternatives need no new labels."""
    for case in cases:
        for group in case.required_evidence_groups_all_of:
            if group.mapping_status in {"unmapped", "parent_only"}:
                continue
            valid = bind_anchor(group.document_version_id, group.source_anchors, snapshot.parents, snapshot.children)
            if not set(group.parent_ids_any_of) <= set(valid["parent_ids_any_of"]):
                raise ValueError("Parent sufficiency: " + group.fact_id)
            # Re-evaluate each sufficient alternative in isolation; do not replace its annotation.
            for ids in group.child_evidence_sets_any_of:
                members = [c for c in snapshot.children if c["child_id"] in ids]
                proof = bind_anchor(group.document_version_id, group.source_anchors, snapshot.parents, members)
                if not any(set(proven) <= set(ids) for proven in proof["child_evidence_sets_any_of"]):
                    raise ValueError("Child sufficiency: " + group.fact_id)


def _load_contents(root):
    snapshot = CorpusSnapshot.model_validate(_json(root / "snapshot.json"))
    cases = load_gold_v2(root / "gold.v2.2.jsonl", snapshot)
    specs = load_gold_answer_specs(root / "answer-spec.jsonl", cases, snapshot)
    return snapshot, cases, specs


def publish_rebound_bundle(*, source_snapshot: Path, source_gold: Path, source_specs: Path, journal_path: Path,
                            target: CorpusSnapshot, evidence: dict[str, Any], output: Path) -> dict[str, Any]:
    sources = {name: path.resolve() for name, path in {"snapshot": source_snapshot, "gold": source_gold,
                                                     "specs": source_specs, "journal": journal_path}.items()}
    if output.exists() or output.is_symlink():
        raise FileExistsError("output exists; never overwrite a frozen bundle")
    if any(output.resolve() == p or output.resolve() in p.parents for p in sources.values()):
        raise ValueError("output overlaps source inputs")
    source_hashes = {k: file_sha256(p) for k, p in sources.items()}
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = {"protocol": "ik-rebound-bundle-v1", "status": "incomplete"}
    _write(output / "bundle-state.json", state)
    try:
        old = CorpusSnapshot.model_validate(_json(source_snapshot))
        old_cases = load_gold_v2(source_gold, old)
        load_gold_answer_specs(source_specs, old_cases, old)
        audit = verify_migration_equivalence(old, target, _json(journal_path), evidence)
        verify_originals(old, old_cases)
        gold_rows = rebind_gold_rows(_rows(source_gold), snapshot_id=target.snapshot_id)
        new_cases = [GoldCaseV2.model_validate(row) for row in gold_rows]
        spec_rows = rebind_spec_rows(_rows(source_specs), cases=new_cases)
        _write(output / "snapshot.json", target.model_dump(mode="json"))
        _write(output / "gold.v2.2.jsonl", gold_rows, rows=True)
        _write(output / "answer-spec.jsonl", spec_rows, rows=True)
        snapshot, cases, _ = _load_contents(output)
        validate_sufficient_sets(cases, snapshot)
        verify_originals(snapshot, cases)
        _write(output / "migration-evidence.json", evidence)
        audit.update(source_files={k: {"path": str(p), "sha256": source_hashes[k]} for k, p in sources.items()},
            counts={"cases": len(cases), "facts": sum(len(c.required_evidence_groups_all_of) for c in cases),
                    "splits": dict(Counter(c.split for c in cases)), "categories": dict(Counter(c.category for c in cases)),
                    "mapping_status": dict(Counter(g.mapping_status for c in cases for g in c.required_evidence_groups_all_of))},
            nonbinding_case_sha256={r["case_id"]: canonical_hash({k: v for k, v in r.items() if k != "snapshot_id"}) for r in gold_rows},
            eligible_for_tuning=False, production_release_accepted=False)
        _write(output / "rebind-audit.json", audit)
        if source_hashes != {k: file_sha256(p) for k, p in sources.items()}:
            raise ValueError("source changed while rebinding")
        _write(output / "bundle-state.json", {**state, "status": "complete",
            "files": {name: file_sha256(output / name) for name in BUNDLE_FILES}})
        load_rebound_bundle(output)
        return audit
    except BaseException as error:
        _write(output / "bundle-state.json", {**state, "error_type": type(error).__name__})
        raise


def load_rebound_bundle(root: Path) -> tuple[CorpusSnapshot, list[GoldCaseV2], dict[str, GoldAnswerSpec], dict[str, Any]]:
    try:
        state = _json(root / "bundle-state.json")
    except FileNotFoundError as error:
        raise ValueError("incomplete bundle") from error
    if state.get("protocol") != "ik-rebound-bundle-v1" or state.get("status") != "complete":
        raise ValueError("incomplete bundle")
    if set(state.get("files", {})) != set(BUNDLE_FILES):
        raise ValueError("bundle file hash manifest incomplete")
    for name, digest in state["files"].items():
        if not (root / name).is_file() or file_sha256(root / name) != digest:
            raise ValueError("bundle hash mismatch: " + name)
    snapshot, cases, specs = _load_contents(root)
    audit = _json(root / "rebind-audit.json")
    if audit.get("verified") is not True or audit.get("target_snapshot_id") != snapshot.snapshot_id:
        raise ValueError("bundle audit not verified")
    for item in audit["source_files"].values():
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("bundle source hash drift")
    return snapshot, cases, specs, audit
