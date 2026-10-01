from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentic_rag.query.router import MemoryContextLoader
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from tests.unit.evals.test_collector import SNAPSHOT


@pytest.mark.asyncio
async def test_disabled_memory_does_not_read():
    snapshot = RuntimeConfigSnapshot.model_validate({**SNAPSHOT.model_dump(), "evaluation": {
        "session_id": "e1", "dataset_sha256": "a" * 64, "corpus_snapshot_id": "b" * 64,
        "memory_policy": "disabled"}})
    memory = SimpleNamespace(load_context=AsyncMock(side_effect=AssertionError("memory read")))
    state = {"runtime_config_snapshot": snapshot.model_dump(mode="json"),
             "scope": {"user_id": "u1"}, "request": {"question": "test"}}
    result = await MemoryContextLoader(memory).load(state)
    memory.load_context.assert_not_awaited()
    assert result["memory_context"]["rendered_context"] == ""


def test_invalid_evaluation_metadata_is_rejected_not_ignored():
    with pytest.raises(ValueError):
        RuntimeConfigSnapshot.model_validate({**SNAPSHOT.model_dump(), "evaluation": {"memory_policy": "read_other_users"}})


@pytest.mark.asyncio
async def test_evaluation_finalizer_skips_write_and_emits_current_snapshot(tmp_path):
    from dataclasses import replace
    from agentic_rag.observability.logging import AgentEventEmitter
    from agentic_rag.persistence.artifacts import LocalArtifactStore
    from agentic_rag.query.graph import build_query_graph
    from agentic_rag.runtime.models import EvaluationMetadata
    from tests.unit.query.test_graph import _deps, _state
    from tests.unit.retrieval.test_graph import RecordingEvents
    events = RecordingEvents()
    state = _state()
    baseline = RuntimeConfigSnapshot.model_validate(state["runtime_config_snapshot"])
    state["runtime_config_snapshot"] = baseline.model_copy(update={"evaluation": EvaluationMetadata(
        session_id="e", dataset_sha256="a" * 64, corpus_snapshot_id="b" * 64)}).model_dump(mode="json")
    memory = SimpleNamespace(load_context=AsyncMock(), extract_and_store=AsyncMock())
    emitter = AgentEventEmitter(events, LocalArtifactStore(tmp_path / "artifacts"),
                                runtime_config_snapshot_id=baseline.snapshot_id)
    deps, *_ = _deps(event_emitter=emitter)
    result = await build_query_graph(replace(deps, memory=memory)).ainvoke(state)
    assert result["termination_reason"] == "completed"
    memory.load_context.assert_not_awaited()
    memory.extract_and_store.assert_not_awaited()
    snapshot_id = RuntimeConfigSnapshot.model_validate(state["runtime_config_snapshot"]).snapshot_id
    assert any(e.event_type == "ANSWER_FINALIZED" for e in events.events)
    assert all(e.runtime_config_snapshot_id == snapshot_id for e in events.events)


@pytest.mark.asyncio
async def test_deployment_mismatch_rejected_before_provider_or_memory():
    from dataclasses import replace
    from agentic_rag.query.graph import build_query_graph
    from tests.unit.query.test_graph import _deps, _state
    deps, *_ = _deps()
    with pytest.raises(ValueError, match="deployment"):
        await build_query_graph(replace(deps, deployment_snapshot_id="different-process-config")).ainvoke(_state())
