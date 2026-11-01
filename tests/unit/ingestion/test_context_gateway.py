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


@pytest.mark.asyncio
async def test_real_gateway_open_circuit_falls_back_without_losing_reservation(
    tmp_path,
):
    from agentic_rag.ingestion.contextualization.service import ContextualizationService
    from agentic_rag.ingestion.contextualization.store import MemoryContextBuildStore
    from tests.unit.ingestion.test_context_service import setup, fence

    client = FakeClient([TimeoutError("offline outage") for _ in range(3)])
    client.max_retries = 0
    adapter = GatewayContextModel(
        client=client,
        snapshot=SNAPSHOT.model_copy(update={"provider_config_fingerprint": "a" * 64}),
        build=config(),
    )
    input = ContextInput(
        target_source_child_id="raw", target_content="body", sources=(), scopes={}
    )
    # Real ModelGateway opens its default circuit after three failed calls.
    for _ in range(3):
        with pytest.raises(TimeoutError):
            await adapter.extract(input)
    assert len(client.responses.calls) == 3

    artifacts, ast, ref, parent, claim = setup(tmp_path, ("本制度适用于星河公司。",))
    store = MemoryContextBuildStore()
    service = ContextualizationService(
        artifacts=artifacts,
        store=store,
        model=adapter,
        count_tokens=len,
        count_model_tokens=lambda text, model_id: len(text),
    )
    result = await service.enrich(
        ast,
        [parent],
        canonical_ref=ref,
        build=config(),
        claim=claim,
        before_side_effect=fence,
    )
    assert (
        result.parents[0].children[0].search_representation.enrichment_status
        == "structural_only"
    )
    assert len(client.responses.calls) == 3
    assert len(store.reservations) == 1
    _, tokens, cost, settled = next(iter(store.reservations.values()))
    assert tokens == config().max_input_tokens + config().max_output_tokens
    assert cost > 0 and not settled
