"""Opt-in live routing acceptance, using actual uploaded and ingested documents."""
import pytest
from uuid import uuid4

pytest_plugins = ("tests.fixtures.routing_services",)
pytestmark = [pytest.mark.e2e, pytest.mark.live_model]


async def query(runtime, question, thread_id=None):
    payload = {"query": question, "wait_seconds": 0, "evaluation": runtime.routing_evaluation}
    payload["thread_id"] = thread_id or f"eval-v2-routing-{uuid4().hex}"
    created = await runtime.client.post("/v1/query", json=payload)
    assert created.status_code == 202, created.text
    run = await runtime.wait_for_terminal(created.json()["run_id"])
    assert run["status"] == "completed"
    events = await runtime.read_sse(run["run_id"])
    return run, events


async def test_weather_one_classification_zero_retrieval_or_research(routing_runtime, monkeypatch):
    calls = []
    gateway = routing_runtime._dependencies.gateway
    original = gateway.complete_structured
    async def observe(call, schema):
        calls.append(schema.__name__)
        return await original(call, schema)
    monkeypatch.setattr(gateway, "complete_structured", observe)
    run, events = await query(routing_runtime, "今天北京天气如何")
    assert calls == ["RouteAssessment"]
    assert run["answer"]["route"] == "chat"
    assert run["answer"]["status"] == "cannot_answer"
    assert not run["answer"].get("evidence_parent_ids")
    assert not ({e["event_type"] for e in events} & {"FAST_RAG_COMPLETED", "RESEARCH_LOOP_COMPLETED"})
    text = str(run["answer"]["segments"])
    assert "尚未接入" in text
    assert "℃" not in text


@pytest.mark.parametrize("question", ["总结我上传的北京天气报告", "刘泽明在京东的工作职责是什么"])
async def test_document_answers_require_audited_citations(routing_runtime, question):
    run, _ = await query(routing_runtime, question)
    assert run["answer"]["route"] in {"fast_rag", "research"}
    assert run["answer"]["audited"] is True
    assert run["answer"]["evidence_parent_ids"]


async def test_comparison_enters_research(routing_runtime):
    run, events = await query(routing_runtime, "比较上传的刘泽明和李四两份简历的项目经验")
    assert run["answer"]["route"] == "research"
    assert any(e["event_type"] == "RESEARCH_LOOP_COMPLETED" for e in events)
    assert run["answer"]["audited"] is True


async def test_real_thread_followup_and_topic_switch(routing_runtime):
    first, _ = await query(routing_runtime, "刘泽明在京东的任职时间是什么")
    second, _ = await query(routing_runtime, "他在那里做了几年", first["thread_id"])
    assert second["answer"]["route"] in {"fast_rag", "research"}
    assert second["answer"]["audited"] is True
    third, events = await query(routing_runtime, "换个话题，今天北京天气如何", first["thread_id"])
    assert third["answer"]["route"] == "chat"
    assert not ({e["event_type"] for e in events} & {"FAST_RAG_COMPLETED", "RESEARCH_LOOP_COMPLETED"})


async def test_mixed_sources_clarify_without_research(routing_runtime):
    run, events = await query(routing_runtime, "结合上传的北京天气报告和今天北京实时天气给出出行建议")
    assert run["answer"]["route"] == "chat"
    assert run["answer"]["status"] == "clarify"
    assert not ({e["event_type"] for e in events} & {"FAST_RAG_COMPLETED", "RESEARCH_LOOP_COMPLETED"})
