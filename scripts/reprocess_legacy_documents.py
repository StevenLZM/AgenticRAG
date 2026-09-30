"""Reprocess frozen legacy originals through the ordinary versioned API.

Never write SQL/ES directly; preserve old artifacts and job IDs on interruption.
"""
import argparse
import asyncio
import json
from pathlib import Path

import httpx

from agentic_rag.config import Settings
from evals.upload import upload_document


async def run(snapshot_path, output_dir):
    settings = Settings()
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot["scope"]["user_id"] != settings.default_user_id:
        raise ValueError("snapshot user scope mismatch")
    selected = [d for d in snapshot["documents"] if d["index_generation"] != settings.index_generation]
    async with httpx.AsyncClient(base_url="http://localhost:8000", trust_env=False, timeout=60) as client:
        response = await client.get("/v1/runtime/summary")
        response.raise_for_status()
        if response.json()["index_generation"] != settings.index_generation:
            raise ValueError("runtime generation mismatch")
        for doc in selected:
            result = await upload_document(client, Path(doc["source_path"]), doc["content_hash"],
                output_dir / (doc["document_id"] + ".json"),
                reprocess_document_id=doc["document_id"], timeout_seconds=900)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["status"] != "completed":
                raise RuntimeError("reprocessing not completed; resume existing job, never duplicate POST")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("var/artifacts/evals/formal-corpus-gold-v2-20260930/snapshot.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("var/artifacts/evals/formal-corpus-gold-v2-20260930-production/reprocessing"))
    args = parser.parse_args()
    asyncio.run(run(args.snapshot, args.output_dir))
