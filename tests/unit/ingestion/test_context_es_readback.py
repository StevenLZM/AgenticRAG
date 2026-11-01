"""Publication readback verifies scoped stored search bytes, not count alone."""

from types import SimpleNamespace
import pytest
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
from agentic_rag.models.contextual import text_sha256
from tests.unit.ingestion.test_manifest import _context


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", [None, "owner", "hash", "id", "partial"])
async def test_context_identity_readback_checks_es_bytes_and_scope(mutation):
    source = dict(
        id="child",
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        representation_version="context-v1",
        contextualized_content="company: 星河公司\n\n500",
        content_hash=text_sha256("company: 星河公司\n\n500"),
    )
    if mutation == "owner":
        source["user_id"] = "user-2"
    if mutation == "hash":
        source["contextualized_content"] = "changed"
    if mutation == "id":
        source["id"] = "other"
    calls = []

    class Client:
        async def search(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                body={
                    "timed_out": mutation == "partial",
                    "_shards": {"failed": 0},
                    "hits": {
                        "total": {"value": 1, "relation": "eq"},
                        "hits": [{"_id": "child", "_source": source}],
                    },
                }
            )

    store = ElasticsearchChildIndexStore(
        Client(), expected_representation_version="context-v1"
    )
    if mutation is None:
        assert await store.child_identities(_context()) == (
            ("child", source["content_hash"]),
        )
    else:
        with pytest.raises(ValueError):
            await store.child_identities(_context())
    assert {"term": {"user_id": "user-1"}} in calls[0]["query"]["bool"]["filter"]
