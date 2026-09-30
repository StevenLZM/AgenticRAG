"""Durable real HTTP uploads. An uncertain POST must be reconciled, not repeated."""
from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
from pathlib import Path
import time
from urllib.parse import quote

import httpx

from evals.report import atomic_write_text

TERMINAL = {"completed", "failed", "quarantined", "rejected"}
JOB_STATES = {"queued", "running", "completed", "failed", "quarantined"}


async def upload_document(client: httpx.AsyncClient, path: Path, sha256: str,
                          ledger_path: Path, *, timeout_seconds=600, poll_interval=2,
                          reprocess_document_id: str | None = None):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("upload bytes differ from frozen corpus")
    binding = {"sha256": sha256, "filename": path.name, "base_url": str(client.base_url)}
    if reprocess_document_id is not None:
        if not reprocess_document_id.strip():
            raise ValueError("missing reprocessing document identity")
        binding["reprocess_document_id"] = reprocess_document_id
    if ledger_path.exists():
        record = json.loads(ledger_path.read_text())
        if (any(record.get(key) != value for key, value in binding.items())
                or record.get("reprocess_document_id") != reprocess_document_id):
            raise ValueError("upload ledger source or endpoint mismatch")
        if record["status"] in TERMINAL:
            return record
        if not record.get("job_id"):
            raise ValueError("reconcile uncertain upload before retry; do not duplicate POST")
    else:
        record = {**binding, "status": "submitting"}

    def persist():
        atomic_write_text(ledger_path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    def apply_job(value):
        if not isinstance(value, dict) or value.get("status") not in JOB_STATES:
            raise ValueError("invalid ingestion response")
        if reprocess_document_id is not None and value.get("document_id") != reprocess_document_id:
            raise ValueError("reprocessing document identity mismatch")
        for key in ("job_id", "document_id", "document_version_id"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError("missing ingestion identity")
            if key in record and record[key] != value[key]:
                raise ValueError("ingestion identity changed")
        record.update({key: value[key] for key in ("job_id", "document_id", "document_version_id", "status")})
        record.pop("error", None)

    if not record.get("job_id"):
        persist()  # Even interruption during the POST leaves an explicit uncertain boundary.
        try:
            endpoint = ("/v1/documents/" + quote(reprocess_document_id, safe="") + "/reprocess"
                        if reprocess_document_id is not None else "/v1/documents")
            response = await client.post(endpoint, files={
                "file": (path.name, payload, mimetypes.guess_type(path.name)[0] or "application/octet-stream")})
            if 400 <= response.status_code < 500:
                record.update(status="rejected", error=f"HTTP {response.status_code}")
            else:
                response.raise_for_status()
                if response.status_code != 202:
                    raise ValueError("unexpected upload status")
                apply_job(response.json())
        except Exception as error:
            record.update(status="uncertain", error=type(error).__name__)
        persist()
        if record["status"] == "uncertain" or record["status"] in TERMINAL:
            return record

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = await client.get("/v1/ingestion-jobs/" + quote(record["job_id"], safe=""))
            response.raise_for_status()
            apply_job(response.json())
        except Exception as error:
            record.update(status="poll_failed", error=type(error).__name__)
            persist()
            return record
        persist()
        if record["status"] in TERMINAL:
            return record
        await asyncio.sleep(poll_interval)
    record.update(status="timeout", error="ingestion deadline exceeded; resume by job_id")
    persist()
    return record
