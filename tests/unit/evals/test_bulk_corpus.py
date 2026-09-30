"""Synthetic corpus contracts; these tests are not real upload evidence."""
from collections import Counter
import importlib


def test_source_has_distinct_connected_documents_and_two_years():
    # Catches accidental duplication, missing formats, and dangling references.
    module = importlib.import_module("evals.bulk_corpus")
    source = module.build_source()
    docs = source["documents"]
    assert len(docs) == len({d["document_key"] for d in docs}) == 1000
    assert Counter(d["format"] for d in docs) == {"txt": 600, "pdf": 200, "xlsx": 200}
    assert {d["year"] for d in docs} == {2025, 2026}
    keys = {d["document_key"] for d in docs}
    for doc in docs:
        assert doc["references"] and set(doc["references"]) <= keys
        assert doc["document_key"] not in doc["references"]
        assert "合成" in doc["notice"]
        assert len(doc["sections"]) >= 4
    policy = next(d for d in docs if d["document_key"] == "S001-2026-policy")
    assert policy["facts"]["lodging_limit_cny_per_night"] == 470
    assert "S001-2025-policy" in policy["references"]


def test_upload_order_starts_with_each_format():
    module = importlib.import_module("evals.bulk_corpus")
    docs = module.build_source()["documents"]
    ordered = module.upload_order(docs)
    assert {d["format"] for d in ordered[:3]} == {"txt", "pdf", "xlsx"}
    assert len({d["document_key"] for d in ordered}) == 1000


def test_manifest_refuses_missing_or_changed_files(tmp_path):
    import pytest
    module = importlib.import_module("evals.bulk_corpus")
    doc = {"document_key": "d", "filename": "d.txt", "format": "txt"}
    with pytest.raises(FileNotFoundError):
        module.freeze_manifest(tmp_path, {"documents": [doc]})
    (tmp_path / "documents").mkdir()
    (tmp_path / "documents/d.txt").write_text("original")
    manifest = module.freeze_manifest(tmp_path, {"documents": [doc]})
    assert manifest["documents"][0]["size_bytes"] == 8
    (tmp_path / "documents/d.txt").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        module.freeze_manifest(tmp_path, {"documents": [doc]})


def test_upload_batch_stops_after_failure_and_resumes_completed(tmp_path):
    import asyncio
    import hashlib
    import httpx
    import pytest
    module = importlib.import_module("scripts.upload_bulk_corpus")
    (tmp_path / "documents").mkdir()
    docs = []
    for n in range(3):
        filename = f"{n}.txt"
        (tmp_path / "documents" / filename).write_bytes(b"data")
        docs.append({"document_key": str(n), "filename": filename, "format": "txt",
                     "sha256": hashlib.sha256(b"data").hexdigest()})
    posted = []
    def transport(request):
        if request.method == "POST":
            n = len(posted)
            posted.append(n)
        else:
            n = int(request.url.path.rsplit("/", 1)[1])
        return httpx.Response(202 if request.method == "POST" else 200, json={
            "job_id": str(n), "document_id": f"d{n}", "document_version_id": f"v{n}",
            "status": "completed" if n == 0 else "failed"})
    async def run():
        async with httpx.AsyncClient(base_url="http://example.test", transport=httpx.MockTransport(transport)) as client:
            with pytest.raises(RuntimeError, match="not completed"):
                await module.upload_batch(client, tmp_path, docs, concurrency=1)
            with pytest.raises(RuntimeError, match="not completed"):
                await module.upload_batch(client, tmp_path, docs, concurrency=1)
    asyncio.run(run())
    assert posted == [0, 1]  # no third upload and no retry of terminal jobs
