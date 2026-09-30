"""HTTP contract tests, not real ingestion evidence."""
import hashlib
import json

import httpx
import pytest

from evals.upload import upload_document


def source(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("synthetic source")
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), tmp_path / "upload.json"


def job(status):
    return {"job_id": "j1", "document_id": "d1", "document_version_id": "v1", "status": status}


@pytest.mark.asyncio
async def test_completed_upload_is_reused_without_post(tmp_path):
    path, digest, ledger = source(tmp_path)
    methods = []

    def transport(request):
        methods.append(request.method)
        return httpx.Response(202 if request.method == "POST" else 200,
                              json=job("queued" if request.method == "POST" else "completed"))

    async with httpx.AsyncClient(base_url="http://isolated.test", transport=httpx.MockTransport(transport)) as client:
        result = await upload_document(client, path, digest, ledger, poll_interval=0)
        assert result["status"] == "completed" and result["document_version_id"] == "v1"
        await upload_document(client, path, digest, ledger)
    assert methods == ["POST", "GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "quarantined"])
async def test_ingestion_failure_is_not_completed(tmp_path, status):
    path, digest, ledger = source(tmp_path)
    async with httpx.AsyncClient(base_url="http://isolated.test", transport=httpx.MockTransport(
        lambda request: httpx.Response(202 if request.method == "POST" else 200, json=job(status))
    )) as client:
        assert (await upload_document(client, path, digest, ledger))["status"] == status


@pytest.mark.asyncio
async def test_rejected_upload_has_no_job_or_success(tmp_path):
    path, digest, ledger = source(tmp_path)
    async with httpx.AsyncClient(base_url="http://isolated.test", transport=httpx.MockTransport(
        lambda request: httpx.Response(415, text="private response")
    )) as client:
        result = await upload_document(client, path, digest, ledger)
    assert result["status"] == "rejected" and "job_id" not in result
    assert "private response" not in ledger.read_text()


@pytest.mark.asyncio
async def test_poll_timeout_resumes_existing_job_without_reupload(tmp_path):
    path, digest, ledger = source(tmp_path)
    state = "running"
    methods = []

    def transport(request):
        methods.append(request.method)
        return httpx.Response(202 if request.method == "POST" else 200, json=job(state))

    async with httpx.AsyncClient(base_url="http://isolated.test", transport=httpx.MockTransport(transport)) as client:
        assert (await upload_document(client, path, digest, ledger, timeout_seconds=0))["status"] == "timeout"
        state = "completed"
        assert (await upload_document(client, path, digest, ledger))["status"] == "completed"
    assert methods.count("POST") == 1


@pytest.mark.asyncio
async def test_uncertain_post_never_automatically_repeats(tmp_path):
    path, digest, ledger = source(tmp_path)

    def transport(request):
        raise httpx.ReadTimeout("response lost")

    async with httpx.AsyncClient(base_url="http://isolated.test", transport=httpx.MockTransport(transport)) as client:
        assert (await upload_document(client, path, digest, ledger))["status"] == "uncertain"
        with pytest.raises(ValueError, match="reconcile"):
            await upload_document(client, path, digest, ledger)
    assert json.loads(ledger.read_text())["status"] == "uncertain"


@pytest.mark.asyncio
async def test_tampered_corpus_rejected_before_network(tmp_path):
    path, digest, ledger = source(tmp_path)
    path.write_text("changed")
    async with httpx.AsyncClient(base_url="http://isolated.test") as client:
        with pytest.raises(ValueError, match="frozen"):
            await upload_document(client, path, digest, ledger)
    assert not ledger.exists()


async def test_reprocessing_ledger_binds_original_document_and_uses_real_endpoint(tmp_path):
    path, digest, ledger = source(tmp_path)
    paths = []
    def transport(request):
        paths.append(request.url.path)
        return httpx.Response(202 if request.method == "POST" else 200,
                              json=job("queued" if request.method == "POST" else "completed"))
    async with httpx.AsyncClient(base_url="http://formal.test", transport=httpx.MockTransport(transport)) as client:
        result = await upload_document(client, path, digest, ledger, reprocess_document_id="d1", poll_interval=0)
        assert result["document_id"] == "d1" and result["status"] == "completed"
        with pytest.raises(ValueError, match="mismatch"):
            await upload_document(client, path, digest, ledger, reprocess_document_id="d2")
        with pytest.raises(ValueError, match="mismatch"):
            await upload_document(client, path, digest, ledger)
    assert paths == ["/v1/documents/d1/reprocess", "/v1/ingestion-jobs/j1"]


async def test_reprocessing_response_cannot_create_another_document(tmp_path):
    path, digest, ledger = source(tmp_path)
    async with httpx.AsyncClient(base_url="http://formal.test", transport=httpx.MockTransport(
        lambda request: httpx.Response(202, json=job("queued"))
    )) as client:
        result = await upload_document(client, path, digest, ledger, reprocess_document_id="d2")
    assert result["status"] == "uncertain" and "job_id" not in result


@pytest.mark.asyncio
async def test_poll_cannot_replace_document_identity(tmp_path):
    path, digest, ledger = source(tmp_path)

    def transport(request):
        data = job("queued" if request.method == "POST" else "completed")
        if request.method == "GET":
            data["document_id"] = "another-document"
        return httpx.Response(202 if request.method == "POST" else 200, json=data)

    async with httpx.AsyncClient(base_url="http://isolated.test", transport=httpx.MockTransport(transport)) as client:
        result = await upload_document(client, path, digest, ledger)
    assert result["status"] == "poll_failed"
    assert result["document_id"] == "d1"
