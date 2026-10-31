import asyncio
from importlib import import_module
from types import SimpleNamespace

import pytest
from sqlalchemy import update

from agentic_rag.runtime.model_gateway import ModelCall, ModelGateway
from tests.unit.orchestration.test_evidence import SNAPSHOT
from tests.unit.orchestration.test_runtime import complete, prepare_and_grade_fast
from agentic_rag.orchestration.contracts import CandidateResult
from tests.unit.orchestration.test_evidence import ranked


class StreamClient:
    def __init__(self, *, incomplete=False):
        self.release = asyncio.Event()
        self.calls = 0
        self.closed = False
        self.incomplete = incomplete
        self.kwargs = None
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        assert kwargs["stream"] is True
        return self

    def __aiter__(self):
        return self.events()

    async def events(self):
        yield {"choices": [{"delta": {"content": "first"}, "finish_reason": None}]}
        await self.release.wait()
        if not self.incomplete:
            yield {"choices": [{"delta": {"content": " second"}, "finish_reason": "stop"}]}
            yield {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}

    async def close(self):
        self.closed = True


@pytest.mark.parametrize("required", [False, True])
def test_omitted_optional_analysis_does_not_invalidate_complete_required_report(required):
    from agentic_rag.orchestration.streaming import missing_required_analysis
    data = {"requires_analysis": True, "tasks": {
        "report": {"spec": {"agent_type": "analysis", "required_for_answer": True}},
        "appendix": {"spec": {"agent_type": "analysis", "required_for_answer": required}},
    }}
    frozen = {"analyses": [{"task_key": "report"}], "limitations": [{"missing_analysis": "appendix", "reason": "answer_context_budget"}]}
    assert missing_required_analysis(data, frozen) is required


async def test_gateway_first_delta_precedes_provider_completion():
    client = StreamClient()
    stream = ModelGateway(client).stream(ModelCall(messages=(), snapshot=SNAPSHOT))
    first = await anext(stream)
    assert first.kind == "delta" and first.text == "first"
    assert not client.release.is_set()
    client.release.set()
    tail = [event async for event in stream]
    assert tail[-1].kind == "completed" and tail[-1].output_tokens == 2
    assert client.closed and client.calls == 1


async def test_provider_eof_after_prefix_never_completes_or_retries():
    client = StreamClient(incomplete=True)
    client.release.set()
    values = []
    with pytest.raises(RuntimeError, match="completion|terminal"):
        async for event in ModelGateway(client).stream(ModelCall(messages=(), snapshot=SNAPSHOT)):
            values.append(event)
    assert [v.kind for v in values] == ["delta"]
    assert client.calls == 1


async def test_streaming_uses_explicit_thinking_policy():
    client = StreamClient()
    client.release.set()
    call = ModelCall(messages=(), snapshot=SNAPSHOT).model_copy(update={"thinking": "disabled"})
    _ = [event async for event in ModelGateway(client).stream(call)]
    assert client.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


async def frozen(store):
    await complete(store, "route", {"kind": "fact", "reason": "fact", "requires_calculation": False, "multiple_targets": False})
    await complete(store, "fast_retrieval", CandidateResult(origin_key="fast", origin="fast_pipeline", criterion_ids=("answer",), candidates=(ranked(),)).model_dump(mode="json"))
    await prepare_and_grade_fast(store)
    item = (await store.list_ready("answer"))[0]
    return await store.claim(item["id"], "answer-worker")


async def test_persisted_chunks_replay_and_cancel_fence_old_owner(store):
    module = import_module("agentic_rag.orchestration.streaming")
    claim = await frozen(store)
    await module.append_delta(store, claim, "first")
    await module.append_delta(store, claim, " second")
    restored = type(store)(store.session_factory)
    chunks = await module.read_chunks(restored, "run-1", "user-1", 1, 1)
    assert [c["data"]["text"] for c in chunks] == [" second"]
    with pytest.raises(KeyError):
        await module.read_chunks(restored, "run-1", "foreign", 1, 0)
    await restored.cancel("run-1", "user-1")
    with pytest.raises(import_module("agentic_rag.orchestration.store").StaleClaim):
        await module.append_delta(store, claim, "bad")
    answer = module.public_answer(await store.read_run("run-1"))
    assert answer.text == "first second"
    assert answer.stream_status == "cancelled"
    assert "audited" not in answer.model_dump()


async def test_long_answer_stream_survives_restart_without_the_old_character_cutoff(store):
    module = import_module("agentic_rag.orchestration.streaming")
    claim = await frozen(store)
    parts = ["分析报告内容。" * 2000 for _ in range(6)]
    for part in parts:
        await module.append_delta(store, claim, part)
    restored = type(store)(store.session_factory)
    chunks = await module.read_chunks(restored, "run-1", "user-1", 0, 0)
    assert "".join(c["data"]["text"] for c in chunks if c["event_type"] == "answer.delta") == "".join(parts)
    await module.complete_text(restored, claim)
    state = (await restored.read_run("run-1"))["data"]
    assert state["answer"]["text"] == "".join(parts)
    assert state["answer"]["status"] == "finalizing"


async def test_quality_model_cannot_approve_an_unverified_extra_number(store):
    module = import_module("agentic_rag.orchestration.streaming")
    claim = await frozen(store)
    state = (await store.read_run("run-1"))["data"]
    key = state["answer"]["input"]["evidence"][0]["evidence_id"]
    await module.append_delta(store, claim, f"A derived difference of 10 percentage points [cite:{key}]")
    await module.complete_text(store, claim)
    quality = await store.claim((await store.list_ready("quality"))[0]["id"], "quality")
    answer = quality.state["answer"]
    await module.complete_quality(store, quality, module.QualityResult(input_hash=answer["input_hash"], answer_hash=answer["answer_hash"], faithfulness="passed"))
    answer = (await store.read_run("run-1"))["data"]["answer"]
    assert answer["faithfulness"] == "failed"
    assert any("unsupported numeric" in issue for issue in answer["quality_issues"])


async def test_expired_stream_is_interrupted_without_replaying_prefix(store):
    from datetime import UTC, datetime, timedelta
    module = import_module("agentic_rag.orchestration.streaming")
    schema = import_module("agentic_rag.orchestration.schema")
    claim = await frozen(store)
    await module.append_delta(store, claim, "prefix")
    async with store.session_factory.begin() as db:
        await db.execute(update(schema.work_items).where(schema.work_items.c.id == claim.work_id).values(lease_until=datetime.now(UTC) - timedelta(seconds=60)))
    await store.recover()
    assert not await store.list_ready("answer")
    value = module.public_answer(await store.read_run("run-1"))
    assert value.text == "prefix" and value.stream_status == "interrupted"
    assert value.faithfulness == "unavailable"
async def test_structured_citations_are_published_only_after_a_valid_marker_is_complete(store):
    from agentic_rag.orchestration.streaming import append_delta, read_chunks
    claim = await frozen(store)
    answer = (await store.read_run("run-1"))["data"]["answer"]
    key = answer["input"]["evidence"][0]["evidence_id"]
    await append_delta(store, claim, "A fact [cite:")
    assert [r["event_type"] for r in await read_chunks(store, "run-1", "user-1", 1, 0)] == ["answer.delta"]
    await append_delta(store, claim, key + "] [cite:invented]")
    rows = await read_chunks(store, "run-1", "user-1", 1, 0)
    citations = [r for r in rows if r["event_type"] == "answer.citation"]
    assert [r["data"]["evidence_id"] for r in citations] == [key]
    await append_delta(store, claim, " again [cite:" + key + "]")
    assert len([r for r in await read_chunks(store, "run-1", "user-1", 1, 0) if r["event_type"] == "answer.citation"]) == 1
