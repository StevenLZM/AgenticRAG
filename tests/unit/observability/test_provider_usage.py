from types import SimpleNamespace

import pytest

from agentic_rag.observability.logging import AgentEventEmitter, event_emission_scope
from agentic_rag.observability.provider_usage import meter_create
from agentic_rag.persistence.artifacts import LocalArtifactStore
from tests.unit.observability.test_degradation_events import RecordingEventRepository


@pytest.mark.asyncio
async def test_sdk_requests_have_distinct_ids_and_actual_cache_usage(tmp_path):
    repo, artifacts = RecordingEventRepository(), LocalArtifactStore(tmp_path)
    emitter = AgentEventEmitter(repo, artifacts, runtime_config_snapshot_id="s")
    async def create(**kwargs):
        return SimpleNamespace(model="actual", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_cache_hit_tokens=70))
    call = meter_create(create, "llm")
    for _ in range(2):
        async with event_emission_scope(emitter, "run", "research", user_id="u"):
            await call(model="asked", messages=[{"content": "private input"}])
    assert [e.event_type for e in repo.events] == ["PROVIDER_REQUEST_STARTED", "PROVIDER_REQUEST_COMPLETED"] * 2
    data = [artifacts.read_json(artifacts.describe(e.payload_ref))["attributes"] for e in repo.events]
    assert data[1]["input_tokens"] == 100 and data[1]["cached_input_tokens"] == 70
    assert data[1]["actual_model"] == "actual"
    assert data[0]["provider_request_id"] != data[2]["provider_request_id"]
    assert "private input" not in str(data)


@pytest.mark.asyncio
async def test_failed_request_is_not_a_zero_token_success(tmp_path):
    repo, artifacts = RecordingEventRepository(), LocalArtifactStore(tmp_path)
    emitter = AgentEventEmitter(repo, artifacts, runtime_config_snapshot_id="s")
    async def create(**kwargs):
        raise TimeoutError("secret response")
    async with event_emission_scope(emitter, "run", "research", user_id="u"):
        with pytest.raises(TimeoutError):
            await meter_create(create, "llm")(model="m")
    assert repo.events[-1].event_type == "PROVIDER_REQUEST_FAILED"
    data = artifacts.read_json(artifacts.describe(repo.events[-1].payload_ref))["attributes"]
    assert "input_tokens" not in data and "secret" not in str(data)
