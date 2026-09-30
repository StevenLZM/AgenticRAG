"""Contract tests for shared domain and runtime models."""

from uuid import UUID

import pytest
from pydantic import ValidationError

from agentic_rag.domain.models import (
    DocumentStatus,
    DocumentVersionStatus,
    JobStatus,
    RunStatus,
    UserScope,
)
from agentic_rag.runtime.ids import content_id, new_id
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SNAPSHOT_DATA = {
    "app_version": "0.1.0",
    "graph_version": "graph-v1",
    "prompt_version": "prompt-v1",
    "main_model_id": "deepseek-v4-pro",
    "light_model_id": "deepseek-v4-flash",
    "embedding_model": "text-embedding-v3",
    "embedding_dimensions": 1024,
    "reranker_version": "bge-reranker-v2-m3",
    "retrieval_config_version": "retrieval-v1",
    "index_generation": "index-v1",
    "memory_config_version": "memory-v1",
}


def test_runtime_snapshot_id_is_content_addressed():
    """Changing neither config nor field order keeps its identity stable."""
    left = RuntimeConfigSnapshot(**SNAPSHOT_DATA)
    right = RuntimeConfigSnapshot.model_validate(left.model_dump())

    assert left.snapshot_id == right.snapshot_id


def test_runtime_snapshot_persists_immutable_prompt_content_hashes():
    router_hash = "a" * 64
    snapshot = RuntimeConfigSnapshot(
        **SNAPSHOT_DATA, prompt_hashes={"router_v1": router_hash}
    )
    changed = RuntimeConfigSnapshot(
        **SNAPSHOT_DATA, prompt_hashes={"router_v1": "b" * 64}
    )

    assert snapshot.model_dump()["prompt_hashes"] == (("router_v1", router_hash),)
    assert snapshot.prompt_hash_map == {"router_v1": router_hash}
    assert snapshot.snapshot_id != changed.snapshot_id
    with pytest.raises(ValidationError):
        snapshot.prompt_hashes = {"router_v1": "c" * 64}
    with pytest.raises(TypeError):
        snapshot.prompt_hashes[0] = ("router_v1", "c" * 64)


def test_user_scope_rejects_blank_user():
    """Whitespace-only users cannot cross the scope boundary."""
    with pytest.raises(ValidationError):
        UserScope(user_id=" ")


def test_user_scope_is_immutable_and_normalizes_whitespace():
    """A scope cannot be reassigned after its normalized ID is established."""
    scope = UserScope(user_id=" user-123 ")

    assert scope.user_id == "user-123"
    with pytest.raises(ValidationError):
        scope.user_id = "other-user"


def test_runtime_snapshot_applies_contract_defaults_and_bounds():
    """Runtime execution limits have safe defaults and reject over-limit work."""
    snapshot = RuntimeConfigSnapshot(**SNAPSHOT_DATA)

    assert snapshot.max_research_rounds == 6
    assert RuntimeConfigSnapshot(
        **SNAPSHOT_DATA, max_research_rounds=8
    ).max_research_rounds == 8
    with pytest.raises(ValidationError):
        RuntimeConfigSnapshot(**SNAPSHOT_DATA, max_research_rounds=9)
    with pytest.raises(ValidationError):
        RuntimeConfigSnapshot(**SNAPSHOT_DATA, max_parallel_subagents_per_run=4)


def test_status_enums_expose_the_specified_wire_values():
    """Persisted status values match the shared protocol."""
    assert [status.value for status in RunStatus] == [
        "queued",
        "running",
        "cancel_requested",
        "cancelled",
        "completed",
        "failed",
    ]
    assert [status.value for status in JobStatus] == [
        "queued",
        "running",
        "completed",
        "quarantined",
        "failed",
    ]
    assert [status.value for status in DocumentStatus] == [
        "processing",
        "active",
        "failed",
        "deleted",
    ]
    assert [status.value for status in DocumentVersionStatus] == [
        "uploaded",
        "building",
        "active",
        "quarantined",
        "failed",
        "inactive",
    ]


def test_new_id_returns_a_uuidv7():
    """New resource identifiers use time-sortable UUIDv7 values."""
    assert UUID(new_id()).version == 7


def test_content_id_is_deterministic_and_unambiguous():
    """Length-prefixing prevents distinct part boundaries from colliding."""
    assert content_id("document", "version") == content_id("document", "version")
    assert content_id("document", "version") != content_id("documentversion")
    assert content_id("a", "bc") != content_id("ab", "c")
