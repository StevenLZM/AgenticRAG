"""The formal audit must fail closed on non-production or incomplete corpora."""
import pytest


def test_release_audit_rejects_legacy_or_unpublished_document():
    from evals.formal_pipeline_audit import require_current_document
    valid = {"pipeline_version": "ingestion-v2", "index_generation": "index-v3",
             "parser_version": "docling-v1", "embedding_version": "text-embedding-v3",
             "searchable": True}
    require_current_document(valid)
    for key, value in [("pipeline_version", "ingestion-v1"), ("index_generation", "index-v2"), ("searchable", False)]:
        with pytest.raises(ValueError):
            require_current_document({**valid, key: value})


def test_release_audit_cannot_hide_processing_or_failed_documents():
    from evals.formal_pipeline_audit import require_corpus_complete
    # Failed historical upload attempts are retained and reported, not counted as
    # required active originals. Processing jobs, however, invalidate completion.
    require_corpus_complete([{"status": "active", "count": 1003}, {"status": "failed", "count": 3}], expected_active=1003)
    with pytest.raises(ValueError):
        require_corpus_complete([{"status": "active", "count": 1002}, {"status": "processing", "count": 1}], expected_active=1003)


def test_foreign_user_documents_cannot_fill_missing_scoped_originals():
    from evals.formal_pipeline_audit import require_corpus_complete
    counts = [{"user_id": "u", "status": "active", "count": 1002},
              {"user_id": "foreign", "status": "active", "count": 1}]
    with pytest.raises(ValueError):
        require_corpus_complete(counts, expected_active=1003, user_id="u", document_count=1002)
    with pytest.raises(ValueError):
        require_corpus_complete([{**counts[0], "count": 1003}], expected_active=1003, user_id="u", document_count=1002)
    require_corpus_complete([{**counts[0], "count": 1003}, {"user_id": "foreign", "status": "processing", "count": 1}],
                            expected_active=1003, user_id="u", document_count=1003)


def test_provenance_and_heading_changes_cannot_pass_a_chunk_identity_audit():
    from evals.formal_pipeline_audit import require_published_fields
    expected = {"content": "same text", "heading_path": ["标题"],
                "ast_locator": {"spans": [{"parent_char_from": 0, "parent_char_to": 9}]}}
    require_published_fields(expected, expected, tuple(expected))
    for key, value in [("heading_path", ["other"]), ("ast_locator", {"spans": [{"parent_char_from": 1, "parent_char_to": 9}]})]:
        with pytest.raises(ValueError):
            require_published_fields({**expected, key: value}, expected, tuple(expected))
    import json
    require_published_fields({**expected, "ast_locator": json.dumps(expected["ast_locator"])}, expected, tuple(expected))
