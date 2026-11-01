import pytest
from agentic_rag.ingestion.contextualization.gateway import GatewayContextModel
from agentic_rag.ingestion.contextualization.assembly import ContextInput
from agentic_rag.runtime.model_gateway import StructuredOutputValidationError
from tests.unit.runtime.test_model_gateway import FakeClient, SNAPSHOT
from tests.unit.ingestion.test_context_service import config


@pytest.mark.asyncio
async def test_production_adapter_malformed_json_never_spends_unreserved_repair():
    client = FakeClient(["malformed", "{}"])
    client.max_retries = 0
    adapter = GatewayContextModel(
        client=client,
        snapshot=SNAPSHOT.model_copy(update={"provider_config_fingerprint": "a" * 64}),
        build=config(),
    )
    with pytest.raises(StructuredOutputValidationError):
        await adapter.extract(
            ContextInput(
                target_source_child_id="raw",
                target_content="body",
                sources=(),
                scopes={},
            )
        )
    assert len(client.responses.calls) == 1
    assert client.responses.calls[0]["model"] == "extract-1"


def test_production_adapter_rejects_hidden_sdk_retries():
    client = FakeClient([])
    with pytest.raises(ValueError, match="max_retries"):
        GatewayContextModel(client=client, snapshot=SNAPSHOT, build=config())


def test_runtime_snapshot_binds_context_fingerprint_without_changing_legacy():
    first = SNAPSHOT.snapshot_id
    updated = SNAPSHOT.model_validate(
        {**SNAPSHOT.model_dump(), "context_build_fingerprint": "b" * 64}
    )
    assert updated.snapshot_id != first
    assert SNAPSHOT.model_validate(SNAPSHOT.model_dump()).snapshot_id == first
