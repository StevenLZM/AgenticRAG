"""Read-only reconciliation of bound paid Judge requests, including zero-call resumes."""
from __future__ import annotations

from evals.gold_v2_validation import strict_json


def reconcile_judge_requests(output, binding):
    artifacts = sorted((output / "judges").glob("*/*.json"))
    path = output / "judge-requests.json"
    if not path.exists():
        if artifacts:
            raise ValueError("judge requests ledger missing; reconcile paid artifacts before resuming")
        return []
    saved = strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or saved.get("binding") != binding:
        raise ValueError("judge requests binding changed")
    records = saved.get("requests")
    if not isinstance(records, list):
        raise ValueError("judge requests records malformed")
    for record in records:
        if (not isinstance(record, dict) or record.get("kind") not in {"llm", "embedding"}
                or record.get("status") not in {"running", "completed", "failed"}
                or not isinstance(record.get("operation"), str) or not record["operation"]
                or any(k not in record for k in ("input_tokens", "output_tokens"))):
            raise ValueError("judge requests record malformed")
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            value = record.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("judge requests token count malformed")
        if (record.get("cached_input_tokens") is not None and record["input_tokens"] is not None
                and record["cached_input_tokens"] > record["input_tokens"]):
            raise ValueError("judge requests cached tokens exceed input")
    operations = {r["operation"] for r in records}
    # Successful cached scores must have corresponding paid-call evidence.
    # Failed/running operations may have stopped before the provider call.
    for artifact in artifacts:
        data = strict_json(artifact.read_text(encoding="utf-8"))
        results = data.get("metrics", {}) if artifact.stem == "ragas" else {artifact.stem: data}
        for name, result in results.items():
            if result.get("status") != "available":
                continue
            if name.startswith("factual_correctness_"):
                name = "factual_correctness_f1"  # one shared paid sample
            if artifact.parent.name + ":" + name not in operations:
                raise ValueError("judge requests missing paid operation evidence")
    return records
