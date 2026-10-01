"""Deterministic, unapproved scoring sidecars derived only from frozen gold."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Literal

from pydantic import Field

from evals.gold_v2_models import (CorpusSnapshot, CriticalField, Digest, FrozenModel,
    GoldCaseV2, Identifier, SourceDocument, Text, canonical_hash)
from evals.gold_v2_validation import load_gold_v2, strict_json


class ReferenceFact(FrozenModel):
    fact_id: Identifier
    claim: Text
    source_fact_ids: list[Identifier] = Field(min_length=1)


class GoldAnswerSpec(FrozenModel):
    protocol: Literal["gold_answer_spec_draft_v1"] = "gold_answer_spec_draft_v1"
    case_id: Identifier
    gold_case_sha256: Digest
    snapshot_id: Digest
    question: Text
    reference_answer: Text
    question_conditions: list[dict[str, str | int]]
    required_facts: list[ReferenceFact] = Field(min_length=1)
    optional_facts: list[ReferenceFact]
    source_refs: list[SourceDocument] = Field(min_length=1)
    supporting_facts: list[ReferenceFact] = Field(min_length=1)
    critical_rules: list[CriticalField]
    review_status: Literal["human_review_pending"] = "human_review_pending"
    review_binding: None = None


def build_draft_spec(case: GoldCaseV2) -> GoldAnswerSpec:
    """Clause boundaries are a draft, NOT claimed human-verified atomicity."""
    claims = [s.strip() for s in re.split(r"[，。；;\n]+|(?<!\d),(?!\d)", case.reference_answer) if s.strip()]
    unique = list(dict.fromkeys(claims))
    source_ids = [f.fact_id for f in case.required_evidence_groups_all_of]
    return GoldAnswerSpec(case_id=case.case_id, gold_case_sha256=canonical_hash(case.model_dump(mode="json")),
        snapshot_id=case.snapshot_id, question=case.question, reference_answer=case.reference_answer,
        question_conditions=[f.conditions for f in case.critical_fields if f.conditions],
        required_facts=[ReferenceFact(fact_id=f"ref-{i}", claim=c, source_fact_ids=source_ids) for i, c in enumerate(unique)],
        optional_facts=[], source_refs=case.source_documents,
        supporting_facts=[ReferenceFact(fact_id=f.fact_id, claim=f.claim, source_fact_ids=[f.fact_id])
                          for f in case.required_evidence_groups_all_of], critical_rules=case.critical_fields)


def validate_spec_for_case(spec: GoldAnswerSpec, case: GoldCaseV2):
    # Phase 1 accepts only reproducible drafts. Human edits/approvals need the
    # separately audited revision workflow, not a handwritten status string.
    if spec != build_draft_spec(case):
        raise ValueError("scoring_spec_differs_from_frozen_gold")


def load_gold_answer_specs(path: Path, cases: list[GoldCaseV2], snapshot: CorpusSnapshot) -> dict[str, GoldAnswerSpec]:
    specs = [GoldAnswerSpec.model_validate(strict_json(line)) for line in path.read_text().splitlines() if line.strip()]
    by_id = {s.case_id: s for s in specs}
    if len(by_id) != len(specs):
        raise ValueError("duplicate_scoring_spec")
    if set(by_id) != {c.case_id for c in cases}:
        raise ValueError("missing_or_foreign_scoring_spec")
    for case in cases:
        if case.snapshot_id != snapshot.snapshot_id:
            raise ValueError("scoring_spec_snapshot_mismatch")
        validate_spec_for_case(by_id[case.case_id], case)
    return by_id


def write_draft_specs(path: Path, cases: list[GoldCaseV2]):
    body = "".join(build_draft_spec(c).model_dump_json() + "\n" for c in cases)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(body)
    path.chmod(0o600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    snapshot = CorpusSnapshot.model_validate(strict_json(args.corpus_snapshot.read_text()))
    cases = load_gold_v2(args.dataset, snapshot)
    write_draft_specs(args.output, cases)
    print(f"Saved {len(cases)} frozen draft specs; no human approvals, query calls or corpus writes.")


if __name__ == "__main__":
    main()
