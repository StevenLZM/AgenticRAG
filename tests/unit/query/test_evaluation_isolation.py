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
