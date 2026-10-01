"""Run the opt-in, model-only routing comparison without touching business data."""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import AsyncOpenAI
from agentic_rag.config import Settings
from agentic_rag.runtime.model_gateway import ModelGateway
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from evals.routing import dataset_hash, load_cases, run_routing_eval, score_routing


async def run(args):
    cases = load_cases(args.dataset)
    settings = Settings()
    if settings.deepseek_api_key is None:
        raise ValueError("missing model credentials")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output directory must be empty")
    snapshot = RuntimeConfigSnapshot(app_version="routing-eval-v1", graph_version="classifier-only-v2",
        prompt_version="router-v1-v2", routing_policy_version="routing-v2",
        main_model_id=settings.main_model, light_model_id=settings.light_model,
        deepseek_protocol=settings.deepseek_protocol, embedding_model=settings.embedding_model,
        embedding_dimensions=1024, reranker_version="unused", retrieval_config_version="unused",
        index_generation="unused", memory_config_version="disabled")
    manifest = {"created_at": datetime.now(UTC).isoformat(), "kind": "real-model-classification-only",
        "dataset_sha256": dataset_hash(cases), "dataset_file_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "snapshot": snapshot.model_dump(mode="json"), "repeats": args.repeats, "cases": len(cases),
        "request_time": "2026-10-01T12:00:00+08:00", "capabilities": {"knowledge_base": True, "external_realtime": False, "external_lookup": False},
        "input_parity": "Both variants receive identical history/time/capabilities; v1 prompt does not instruct how to use new fields.",
        "client_timeout_seconds": 30, "sdk_retries": 0, "gateway_retries": 2, "concurrency": 1}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    rows = []
    with (output / "samples.jsonl").open("x") as stream:
        def record(sample):
            rows.append(sample)
            stream.write(sample.model_dump_json() + "\n")
            stream.flush()
            print(f"{len(rows)}/{len(cases) * args.repeats * 2} {sample.case_id} {sample.variant} {sample.actual_route or sample.error_code}", flush=True)
        try:
            async with AsyncOpenAI(api_key=settings.deepseek_api_key.get_secret_value(),
                base_url=settings.deepseek_base_url, timeout=30, max_retries=0) as client:
                await run_routing_eval(cases, ModelGateway(client), snapshot, repeats=args.repeats, on_sample=record)
        finally:
            report = score_routing(rows, cases, repeats=args.repeats)
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            (output / "report.md").write_text("# Routing evaluation\n\nStatus: " + report["status"] +
                "\n\nClassification only; not an end-to-end ingestion/retrieval benchmark.\n\n```json\n" +
                json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n")
    print(report["status"], flush=True)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[report["status"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/datasets/routing_v2.jsonl"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    try:
        raise SystemExit(asyncio.run(run(parser.parse_args())))
    except (ValueError, OSError):
        print("INCOMPLETE: invalid configuration, dataset or output directory", file=sys.stderr)
        raise SystemExit(2)
