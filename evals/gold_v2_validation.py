"""Fail closed on malformed, cross-version, or arithmetically incorrect gold."""
from __future__ import annotations

import json
import re
from pathlib import Path

from evals.gold_v2_models import CorpusSnapshot, GoldCaseV2, finite_number


def strict_json(text: str):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("non-finite JSON number: " + value)

    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def validate_derivation(case: GoldCaseV2):
    if case.derivation is None:
        return
    operation, inputs = case.derivation.operation, case.derivation.inputs
    if operation == "2026_limit-2025_limit":
        expected = finite_number(inputs["2026"]) - finite_number(inputs["2025"])
        # Arithmetic alone does not bind the operands to original evidence.
        # Anchors are separately verified against the hashed originals.
        for value in inputs.values():
            pattern = r"(?<![\d.])" + re.escape(format(finite_number(value), "g")) + r"(?![\d.])"
            if not any(re.search(pattern, anchor) for group in case.required_evidence_groups_all_of for anchor in group.source_anchors):
                raise ValueError("calculation input has no matching source anchor")
    elif operation == "B6+C6-D6":
        # Frozen inputs are the original A6:D6 row: A is the inventory key.
        expected = finite_number(inputs[1]) + finite_number(inputs[2]) - finite_number(inputs[3])
    elif operation == "(1-sum(C6:C9)/sum(B6:B9))*100":
        denominator = sum(finite_number(row[1]) for row in inputs)
        if denominator <= 0:
            raise ValueError("derivation denominator must be positive")
        expected = (1 - sum(finite_number(row[2]) for row in inputs) / denominator) * 100
    else:
        raise ValueError("unsupported gold derivation; implement an independent verifier first")
    if len(case.critical_fields) != 1 or abs(case.critical_fields[0].value - expected) > case.critical_fields[0].tolerance + 1e-10:
        raise ValueError("gold derivation does not equal expected critical field")


def load_gold_v2(path: Path, snapshot: CorpusSnapshot) -> list[GoldCaseV2]:
    docs = {(d["document_id"], d["document_version_id"]): d for d in snapshot.documents}
    parents = {p["id"]: p for p in snapshot.parents}
    children = {c["child_id"]: c for c in snapshot.children}
    cases, seen = [], set()
    for ordinal, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            case = GoldCaseV2.model_validate(strict_json(line))
            if case.case_id in seen:
                raise ValueError("duplicate case_id")
            if case.snapshot_id != snapshot.snapshot_id or case.user_id != snapshot.scope["user_id"]:
                raise ValueError("gold belongs to a different corpus/user")
            versions = set()
            for source in case.source_documents:
                document = docs.get((source.document_id, source.document_version_id))
                if document is None or document["content_hash"] != source.sha256:
                    raise ValueError("gold source version/hash mismatch")
                versions.add(source.document_version_id)
            for group in case.required_evidence_groups_all_of:
                if group.document_version_id not in versions:
                    raise ValueError("fact has no original source")
                for ids, members in ((group.parent_ids_any_of, parents),
                                     ([c for ids in group.child_evidence_sets_any_of for c in ids], children)):
                    for chunk_id in ids:
                        item = members.get(chunk_id)
                        if item is None or item["user_id"] != case.user_id or item["document_version_id"] != group.document_version_id:
                            raise ValueError("foreign/missing gold chunk")
            for qrel in case.graded_qrels:
                item = (children if qrel.level == "child" else parents).get(qrel.chunk_id)
                if item is None or item["user_id"] != case.user_id:
                    raise ValueError("graded qrel outside snapshot")
            validate_derivation(case)
            seen.add(case.case_id)
            cases.append(case)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError(f"gold row {ordinal}: {error}") from error
    if not cases:
        raise ValueError("empty gold dataset")
    # Prevent company/family leakage without inventing an equivalence annotation.
    assignments = {}
    for case in cases:
        keys = {s.document_version_id for s in case.source_documents}
        if case.question_family:
            keys.add("family:" + case.question_family)
        if case.case_id.startswith("S") and "-" in case.case_id:
            keys.add("company:" + case.case_id.split("-", 1)[0])
        for key in keys:
            if key in assignments and assignments[key] != case.split:
                raise ValueError("gold company/source/question family crosses data splits")
            assignments[key] = case.split
    return cases
