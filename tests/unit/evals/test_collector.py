from copy import deepcopy

import pytest

from evals.collector import project_verified_state, read_checkpoint, RuntimeCollector
from agentic_rag.runtime.models import RuntimeConfigSnapshot

SNAPSHOT = RuntimeConfigSnapshot(app_version="test", graph_version="test", prompt_version="test",
    main_model_id="test", light_model_id="test", embedding_model="test", embedding_dimensions=1024,
    reranker_version="test", retrieval_config_version="test", index_generation="g1", memory_config_version="test")


def observed():
    run = {"id": "run1", "user_id": "u1", "runtime_config_snapshot_id": SNAPSHOT.snapshot_id, "question": "question",
           "status": "completed", "answer": {"audited": True}, "route": "research"}
    trace = {"schema_version": 1, "user_id": "u1", "snapshot_id": SNAPSHOT.snapshot_id, "index_generation": "g1",
             "stages": {name: [{"child_id": "c1", "parent_id": "p1", "user_id": "u1",
                                "document_id": "d1", "document_version_id": "v1"}]
                        for name in ("dense", "bm25", "rrf", "rerank")},
             "selected_parent_ids": ["p1"], "hydrated_parent_ids": ["p1"]}
    state = {"run_id": "run1", "scope": {"user_id": "u1"}, "request": {"question": "question"},
             "runtime_config_snapshot": SNAPSHOT.model_dump(mode="json"),
             "route": {"route": "research"}, "answer": {"audited": True},
             "retrieval_batches": [{"observation": trace, "parents": [{"parent_id": "p1"}], "target_ids": ["todo1"]}],
             "packed_context": {"index_generation": "g1", "rendered_context": "actual packed text",
                                "items": [{"parent_id": "p1", "content": "actual evidence"}]}}
    segment = {"kind": "content", "text": "answer", "evidence_ids": []}
    state["answer"] = {"audited": True, "segments": [segment]}
    state["termination_reason"] = "completed"
    run["answer"] = {"audited": True, "segments": [segment], "route": "research",
                     "runtime_config_snapshot_id": SNAPSHOT.snapshot_id}
    return run, state


def collect(run, state):
    return project_verified_state(run, state, run_id="run1", user_id="u1", snapshot_id=SNAPSHOT.snapshot_id, question="question")


def test_observed_route_context_and_rank_not_expected_or_citation():
    result = collect(*observed())
    assert result["route"] == "research"
    assert result["contexts"] == ["actual evidence"]
    assert result["ranked_parent_ids"] == ["p1"]
    assert result["retrieval_rounds"][0]["stages"]["dense"] == ["p1"]


@pytest.mark.parametrize("boundary", ["run", "state", "trace", "missing_trace"])
def test_foreign_or_missing_observation_is_rejected(boundary):
    run, state = observed()
    if boundary == "run":
        run["user_id"] = "other"
    elif boundary == "state":
        state["runtime_config_snapshot"]["graph_version"] = "old"
    elif boundary == "trace":
        state["retrieval_batches"][0]["observation"]["user_id"] = "other"
    else:
        state["retrieval_batches"][0].pop("observation")
    with pytest.raises(ValueError):
        collect(run, state)


def test_multiple_retrievals_are_not_concatenated_into_one_ranking():
    run, state = observed()
    state["retrieval_batches"].append(deepcopy(state["retrieval_batches"][0]))
    result = collect(run, state)
    assert result["ranked_parent_ids"] is None
    assert len(result["retrieval_rounds"]) == 2
    assert result["ranking_scope"] == "per-retrieval"


def test_persisted_public_answer_matches_projected_checkpoint_not_raw_dict():
    run, state = observed()
    segment = {"kind": "content", "text": "answer", "evidence_ids": []}
    state["answer"] = {"audited": True, "segments": [segment]}
    state["termination_reason"] = "completed"
    run["route"] = None  # SQL route column is currently not populated by worker.
    run["answer"] = {"audited": True, "segments": [segment], "route": "research",
                     "runtime_config_snapshot_id": SNAPSHOT.snapshot_id}
    assert collect(run, state)["route"] == "research"


def test_checkpoint_reader_is_read_only_and_scoped(tmp_path):
    import sqlite3
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    path = tmp_path / "checkpoint.sqlite"
    kind, payload = JsonPlusSerializer().dumps_typed({"channel_values": {"run_id": "run1"}})
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE checkpoints(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, type TEXT, checkpoint BLOB)")
        db.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?)", ("query:u1:t1", "", "1", kind, payload))
    before = path.read_bytes()
    assert read_checkpoint(path, "query:u1:t1")["run_id"] == "run1"
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="checkpoint"):
        read_checkpoint(path, "query:u2:t1")
    assert not (tmp_path / "missing.sqlite").exists()


@pytest.mark.asyncio
async def test_collector_rejects_foreign_run_before_loading_checkpoint(tmp_path):
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as db:
        await db.execute(text("CREATE TABLE agent_runs(id TEXT,user_id TEXT,runtime_config_snapshot_id TEXT,"
                              "question TEXT,status TEXT,answer JSON,route TEXT,checkpoint_thread_id TEXT)"))
        await db.execute(text("INSERT INTO agent_runs(id,user_id) VALUES('run1','foreign')"))
    collector = RuntimeCollector(engine, tmp_path / "missing.sqlite")
    try:
        with pytest.raises(ValueError, match="scoped run"):
            await collector.collect(run_id="run1", user_id="u1", snapshot_id=SNAPSHOT.snapshot_id, question="question")
    finally:
        await engine.dispose()
    with pytest.raises(Exception):
        read_checkpoint(tmp_path / "missing.sqlite", "query:u1:t1")
    assert not (tmp_path / "missing.sqlite").exists()
