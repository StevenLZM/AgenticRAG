from types import SimpleNamespace

import pytest

from evals.real_stack import isolated_environment


def settings():
    return SimpleNamespace(mysql_dsn="mysql+asyncmy://user:secret@localhost/business",
                           redis_url="redis://localhost:6379/0", elasticsearch_url="http://localhost:9200")


def allocation():
    return {"allocation_id": "abc123", "mysql_database": "agentic_rag_eval_abc123",
            "redis_database_candidate": 1, "elasticsearch_url": "http://127.0.0.1:9201",
            "user_id": "rag-eval-abc123", "index_generation": "eval-abc123",
            "mem0_collection": "agent_memories_eval_abc123",
            "artifact_root": "var/artifacts/evals/example/runtime"}


def test_environment_isolates_all_persistent_boundaries(tmp_path):
    env = isolated_environment(settings(), allocation(), root=tmp_path)
    assert env["AGENTIC_RAG_MYSQL_DSN"].endswith("/agentic_rag_eval_abc123")
    assert env["AGENTIC_RAG_REDIS_URL"] == "redis://localhost:6379/1"
    assert env["AGENTIC_RAG_ELASTICSEARCH_URL"] == "http://127.0.0.1:9201"
    assert env["AGENTIC_RAG_DEFAULT_USER_ID"] == "rag-eval-abc123"
    for name in ("ARTIFACT_ROOT", "QUERY_CHECKPOINT_PATH", "INGESTION_CHECKPOINT_PATH", "MEM0_HISTORY_DB_PATH"):
        assert env["AGENTIC_RAG_" + name].startswith(str(tmp_path / "var/artifacts/evals/example/runtime"))


@pytest.mark.parametrize("change", [
    {"mysql_database": "business"}, {"redis_database_candidate": 0},
    {"elasticsearch_url": "http://127.0.0.1:9200"}, {"artifact_root": "../outside"},
    {"mem0_collection": "agent_memories_v1"},
])
def test_unsafe_allocation_is_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        isolated_environment(settings(), {**allocation(), **change}, root=tmp_path)
