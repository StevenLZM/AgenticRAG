"""Fail-closed gates for the normal, fully published production corpus."""
import json


def require_current_document(doc):
    required = {"pipeline_version": "ingestion-v2", "index_generation": "index-v3",
                "parser_version": "docling-v1", "embedding_version": "text-embedding-v3",
                "searchable": True}
    if any(doc.get(k) != value for k, value in required.items()):
        raise ValueError("document is not fully published through the current production pipeline")


def require_corpus_complete(status_counts, *, expected_active, user_id=None, document_count=None):
    if user_id is not None:
        status_counts = [r for r in status_counts if r.get("user_id") == user_id]
    active = sum(r["count"] for r in status_counts if r["status"] == "active")
    processing = sum(r["count"] for r in status_counts if r["status"] == "processing")
    if processing or active != expected_active or (document_count is not None and document_count != expected_active):
        raise ValueError("required corpus has missing or processing documents")


def require_published_fields(actual, expected, fields):
    for field in fields:
        if field not in actual or field not in expected:
            raise ValueError("missing published chunk metadata: " + field)
        value = actual[field]
        if field == "ast_locator" and isinstance(value, str):
            value = json.loads(value)
        if value != expected[field]:
            raise ValueError("published chunk metadata differs: " + field)
