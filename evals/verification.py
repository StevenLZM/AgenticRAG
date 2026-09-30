"""Recheck scored rows against live scoped runtime state, including on resume."""
import hashlib
import json
from pathlib import Path

from evals.report import atomic_write_text


def verify_projection(observed, row):
    from evals.run import _response_answer

    if _response_answer(observed) != row.answer:
        raise ValueError("scored answer differs from verified runtime")
    if observed["route"] != row.route or observed["ranking_scope"] != row.ranking_scope:
        raise ValueError("scored route or ranking scope differs from runtime")
    if tuple(observed["contexts"]) != row.retrieved_contexts:
        raise ValueError("scored contexts differ from runtime")
    ranks = [{key: value[key] for key in ("ordinal", "target_ids", "stages", "parent_ids")}
             for value in observed["retrieval_rounds"]]
    if ranks != [r.model_dump(mode="json") for r in row.retrieval_rounds]:
        raise ValueError("scored rankings differ from runtime")


async def verify_runtime_rows(cases, rows, client, output_dir):
    from evals.clients import HttpQueryClient
    from evals.collector import RuntimeCollector

    # Graph/contract clients and arbitrary provenance labels are not evidence.
    if not isinstance(client, HttpQueryClient) or not isinstance(client._collector, RuntimeCollector):
        return []
    receipts = []
    for ordinal, (case, row) in enumerate(zip(cases, rows, strict=True)):
        ledger = json.loads((client._ledger_dir / f"{case.case_id}.json").read_text())
        fingerprint = hashlib.sha256(case.model_dump_json().encode()).hexdigest()
        if ledger["case_sha256"] != fingerprint or ledger["base_url"] != client._base_url:
            raise ValueError("runtime ledger binding mismatch")
        observed = await client._collector.collect(run_id=ledger["run_id"], user_id=case.user_id,
                                                   snapshot_id=case.runtime_config_snapshot_id, question=case.question)
        verify_projection(observed, row)
        from evals.run import _deterministic_metrics
        rows[ordinal] = row.model_copy(update={"deterministic_metrics": _deterministic_metrics(
            case, ranked_parent_ids=observed["ranked_parent_ids"] or [], response=observed, events=[]
        )})
        encoded = json.dumps(observed, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        relative = f"runtime-evidence/{case.case_id}.json"
        atomic_write_text(output_dir / relative, encoded)
        receipts.append({"case_id": case.case_id, "run_id": ledger["run_id"],
                         "snapshot_id": case.runtime_config_snapshot_id, "artifact": relative,
                         "sha256": hashlib.sha256(encoded.encode()).hexdigest()})
    if len({r["run_id"] for r in receipts}) != len(receipts):
        raise ValueError("distinct cases reused the same runtime run")
    return receipts


async def verify_saved_evaluation(run_dir: Path) -> dict:
    """Read-only verification; never submit queries, upload, or call a judge."""
    from evals.saved_report import verify_saved_evaluation as verify
    return await verify(Path(run_dir))
