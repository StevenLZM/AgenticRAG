"""Real judge connectivity check; never a RAG quality evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evals.report import atomic_write_text

SAMPLES = (
    dict(question="星岚科技每周允许几天远程办公？", answer="每周允许两天远程办公。",
         contexts=["星岚科技允许员工每周远程办公两天。"],
         reference_answer="每周两天。", answerable=True),
    dict(question="星岚科技明年的营业收入是多少？", answer="现有资料没有明年的营业收入，无法确定。",
         contexts=[], reference_answer="资料不足，无法确定。", answerable=False),
)


async def run_preflight(judge, output_path: Path, *, retry=False):
    output_path = Path(output_path)
    if output_path.exists():
        record = json.loads(output_path.read_text())
        if record.get("judge_fingerprint") != judge.fingerprint:
            raise ValueError("judge changed; use a new output path")
        if record.get("status") == "available":
            if record.get("schema_version") != 1 or len(record.get("samples", [])) != len(SAMPLES):
                raise ValueError("invalid completed preflight record")
            return record
        if not retry:
            raise ValueError("incomplete calls require explicit retry")
    else:
        record = {"schema_version": 1, "purpose": "judge-connectivity-only",
                  "real_query_count": 0, "judge_fingerprint": judge.fingerprint,
                  "judge_metadata": judge.metadata, "samples": []}
    record["status"] = "running"
    rows = record.setdefault("samples", [])

    def persist():
        atomic_write_text(output_path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    persist()
    for ordinal, sample in enumerate(SAMPLES):
        if ordinal < len(rows) and rows[ordinal].get("status") == "available":
            continue
        if ordinal == len(rows):
            rows.append({"attempts": []})
        row = rows[ordinal]
        row["status"] = "running"
        attempt = {"status": "running"}
        row.setdefault("attempts", []).append(attempt)
        persist()
        try:
            result = (await judge.evaluate(**sample)).as_dict()
        except Exception as error:
            result = {"status": "failed", "metrics": {}, "reason": type(error).__name__}
        attempt.update(result)
        row.update(result)
        persist()
    record["status"] = "available" if all(r["status"] == "available" for r in rows) else "failed"
    persist()
    return record


async def _main(args):
    from agentic_rag.config import Settings
    from evals.judge import JudgeConfig, RagasJudge

    judge = RagasJudge(JudgeConfig.from_settings(Settings()))
    try:
        record = await run_preflight(judge, args.output, retry=args.retry_incomplete)
        print(json.dumps({"output": str(args.output), "status": record["status"],
                          "purpose": record["purpose"], "real_query_count": 0}))
        return 0 if record["status"] == "available" else 1
    finally:
        await judge.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-incomplete", action="store_true")
    raise SystemExit(asyncio.run(_main(parser.parse_args())))
