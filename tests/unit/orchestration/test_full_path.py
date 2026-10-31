"""The composed stage chain runs with deterministic providers and real SQL."""
from collections import defaultdict
from types import SimpleNamespace

from agentic_rag.orchestration.execution import StageExecutor
from agentic_rag.orchestration.agents import AnalysisDraft, QueryPlan
from agentic_rag.orchestration.evidence import EvidencePreparer
from agentic_rag.orchestration.service import CoordinationService
from agentic_rag.orchestration.worker import AgentWorker
from agentic_rag.runtime.model_gateway import ModelResponse
from agentic_rag.runtime.model_stream import ModelStreamEvent
from tests.unit.orchestration.test_store import plan  # noqa: F401
from tests.unit.orchestration.test_agents import analysis_task
from tests.unit.orchestration.test_evidence import Hydrator, ranked


def _schema_name(schema):
    for base in (AnalysisDraft, QueryPlan):
        if issubclass(schema, base):
            return base.__name__
    return schema.__name__


class Broker:
    def __init__(self):
        self.queues = defaultdict(list)
        self.seq = 0
    async def publish(self, stream, aggregate_id, enqueued_at, dedupe_key=None):
        self.seq += 1
        self.queues[stream].append(SimpleNamespace(id=str(self.seq), aggregate_id=aggregate_id))
    async def consume(self, stream, group, consumer, block_ms):
        messages, self.queues[stream] = self.queues[stream], []
        return messages
    async def reclaim(self, *args): return []
    async def ack(self, *args): pass


class Gateway:
    def __init__(self):
        self.calls = []
    async def complete_structured(self, call, schema):
        import json
        data = json.loads(call.messages[1]["content"])
        self.calls.append(_schema_name(schema))
        name = _schema_name(schema)
        if name == "RouteDecision":
            value = {"kind": "analysis", "reason": "growth report", "requires_calculation": True, "multiple_targets": True}
        elif name == "SupervisorDecision":
            if not data["tasks"]:
                raw = plan().model_dump(mode="json")
                for task in raw["tasks"]:
                    task["sources"]["document_ids"] = []
                a = analysis_task().model_dump(mode="json")
                a["sources"]["document_ids"] = []
                a["dependencies"] = [{"task_key": "r1"}, {"task_key": "r2"}]
                raw["tasks"].append(a)
                value = {"action": "dispatch", "reason": "parallel sources then analysis", "plan": raw}
            else:
                value = {"action": "answer", "reason": "all accepted"}
        elif name == "QueryPlan":
            value = {"queries": [data["task"]["key"] + " revenue"], "reason": "source guidance"}
        elif name == "CandidateSummary":
            value = {"findings": [{"statement": "Revenue source", "evidence_refs": [data["candidates"][0]["child_id"]], "criterion_ids": ["revenue"]}]}
        elif name == "GradeResult":
            value = {"bundle_hash": data["bundle_hash"], "decision": "sufficient", "coverage": [{"criterion_id": c["id"], "status": "supported", "evidence_refs": data["evidence_ids"]} for c in data["criteria"]]}
        elif name == "AnalysisDraft":
            b = data["inputs"][0]["bundle"]
            from agentic_rag.orchestration.contracts import PreparedBundle
            item = b["packed"]["items"][0]
            operands = [{"value": value, "evidence_ref": item["evidence_id"], "bundle_hash": PreparedBundle.model_validate(b).bundle_hash,
                         "quote": item["content"], "unit": "million", "entity": "ACME", "period": str(year)} for value, year in ((100, 2024), (120, 2025))]
            value = {"calculations": [{"id": "growth", "operation": "growth_percent", "operands": operands, "criterion_ids": ["revenue"]}]}
        elif name == "AnalysisAssessment":
            value = {"result_hash": data["result_hash"], "decision": "accepted", "supported_criteria": ["revenue"]}
        elif name == "QualityResult":
            value = {"input_hash": data["input_hash"], "answer_hash": data["answer_hash"], "faithfulness": "passed"}
        else:
            raise AssertionError(name)
        return ModelResponse(value=schema.model_validate(value), requested_model="test", actual_model="test", input_tokens=10, output_tokens=20, attempts=1, latency_ms=1)
    async def stream(self, call):
        import json
        assert call.max_output_tokens == 100000
        frozen = json.loads(call.messages[1]["content"])
        assert frozen["analyses"][0]["result"]["calculations"][0]["value"] == 20
        yield ModelStreamEvent(kind="delta", text="Growth is 20% ")
        yield ModelStreamEvent(kind="delta", text="[cite:" + frozen["evidence"][0]["evidence_id"] + "]")
        yield ModelStreamEvent(kind="completed", output_tokens=20)


class Retrieval:
    async def retrieve_candidates(self, request, scope, snapshot):
        return ranked(content="ACME revenue was 100 million in 2024 and 120 million in 2025.").model_copy(update={"query": request.query})


async def test_parallel_retrieval_analysis_and_stream_complete_after_coordinator_restart(store):
    gateway, broker = Gateway(), Broker()
    executor = StageExecutor(store, gateway, retrieval=Retrieval(), preparer=EvidencePreparer(Hydrator()))
    worker = AgentWorker(store, executor, worker_id="worker", role="all")
    for tick in range(30):
        # New coordinator/store each tick proves no causal state lives in RAM.
        restored = type(store)(store.session_factory)
        await CoordinationService(restored, broker, worker_id=f"coordinator-{tick}").tick(public_runs=False)
        await worker.run_one()
        state = await store.read_run("run-1")
        if state["status"] == "completed":
            break
    assert state["status"] == "completed", state["data"]
    assert state["data"]["answer"]["faithfulness"] == "passed"
    assert set(state["data"]["tasks"]) == {"r1", "r2", "a1"}
    assert gateway.calls.count("SupervisorDecision") == 2
    assert gateway.calls.index("GradeResult") < gateway.calls.index("AnalysisDraft") < gateway.calls.index("AnalysisAssessment")
    assert state["data"]["tasks"]["a1"]["acceptance"] == "accepted"


async def test_user_arithmetic_runs_analysis_assessment_and_stream_without_retrieval(store):
    import json
    from agentic_rag.orchestration.contracts import content_hash
    from agentic_rag.orchestration.store_sql import lock_run, save_run
    question = "Calculate 100 + 120"
    async with store.session_factory.begin() as db:
        run = await lock_run(db, "run-1")
        run["data"]["question"] = question
        await save_run(db, run)
    class ArithmeticGateway(Gateway):
        async def complete_structured(self, call, schema):
            data = json.loads(call.messages[1]["content"])
            if schema.__name__ == "SupervisorDecision" and not data["tasks"]:
                raw = analysis_task().model_dump(mode="json")
                value = {"action": "dispatch", "reason": "user supplied operands", "plan": {"tasks": [raw]}}
            elif issubclass(schema, AnalysisDraft):
                operands = [{"value": n, "source_kind": "user_input", "evidence_ref": "query", "bundle_hash": content_hash({"question": question}),
                             "quote": question, "unit": "number", "entity": "user_input", "period": "unspecified"} for n in (100, 120)]
                value = {"calculations": [{"id": "total", "operation": "sum", "operands": operands, "criterion_ids": ["revenue"]}]}
            else:
                return await super().complete_structured(call, schema)
            self.calls.append(_schema_name(schema))
            return ModelResponse(value=schema.model_validate(value), requested_model="test", actual_model="test", input_tokens=10, output_tokens=20, attempts=1, latency_ms=1)
        async def stream(self, call):
            frozen = json.loads(call.messages[1]["content"])
            assert frozen["evidence"] == []
            assert frozen["analyses"][0]["result"]["calculations"][0]["value"] == 220
            yield ModelStreamEvent(kind="delta", text="100 + 120 = 220")
            yield ModelStreamEvent(kind="completed", output_tokens=10)
    gateway, broker = ArithmeticGateway(), Broker()
    worker = AgentWorker(store, StageExecutor(store, gateway), worker_id="calculator", role="all")
    for tick in range(20):
        await CoordinationService(store, broker, worker_id=str(tick)).tick(public_runs=False)
        await worker.run_one()
        state = await store.read_run("run-1")
        if state["status"] == "completed":
            break
    assert state["status"] == "completed", state["data"]
    assert state["data"]["answer"]["faithfulness"] == "passed"
    assert set(state["data"]["tasks"]) == {"a1"}
    assert "QueryPlan" not in gateway.calls and "GradeResult" not in gateway.calls


async def test_quality_model_cannot_approve_answer_when_required_analysis_failed(store):
    import json
    class UnavailableAnalysis(Gateway):
        async def complete_structured(self, call, schema):
            data = json.loads(call.messages[1]["content"])
            if schema.__name__ == "SupervisorDecision":
                value = ({"action": "dispatch", "reason": "analyze", "plan": {"tasks": [analysis_task().model_dump(mode="json")]}}
                         if not data["tasks"] else {"action": "answer", "answer_mode": "partial", "reason": "analysis failed"})
                return ModelResponse(value=schema.model_validate(value), requested_model="test", actual_model="test", input_tokens=10, output_tokens=20, attempts=1, latency_ms=1)
            if issubclass(schema, AnalysisDraft):
                raise ValueError("operand quote does not match original input")
            return await super().complete_structured(call, schema)
        async def stream(self, call):
            assert json.loads(call.messages[1]["content"])["analyses"] == []
            yield ModelStreamEvent(kind="delta", text="100 + 120 = 220")
            yield ModelStreamEvent(kind="completed", output_tokens=10)
    gateway, broker = UnavailableAnalysis(), Broker()
    worker = AgentWorker(store, StageExecutor(store, gateway), worker_id="calculator", role="all")
    for tick in range(20):
        await CoordinationService(store, broker, worker_id=str(tick)).tick(public_runs=False)
        await worker.run_one()
        state = await store.read_run("run-1")
        if state["status"] == "completed":
            break
    assert state["status"] == "completed"
    assert state["data"]["answer"]["input"]["mode"] == "partial"
    assert state["data"]["answer"]["faithfulness"] == "unavailable"
    assert any("analysis" in issue for issue in state["data"]["answer"]["quality_issues"])


async def test_delayed_analysis_assessment_cannot_reaccept_superseded_inputs(store):
    from agentic_rag.orchestration.coordinator import Coordinator
    executor = StageExecutor(store, Gateway(), retrieval=Retrieval(), preparer=EvidencePreparer(Hydrator()))
    coordinator = Coordinator(store)
    for stage in ("route", "supervisor", "retrieval", "retrieval", "prepare", "grade", "analysis"):
        work = (await store.list_ready(stage))[0]
        claim = await store.claim(work["id"], stage)
        result = await store.submit_result(claim, await executor(claim))
        await store.receive_result(result)
        await coordinator.advance("run-1")
    work = (await store.list_ready("analysis_grade"))[0]
    claim = await store.claim(work["id"], "delayed-assessor")
    old_assessment = await store.submit_result(claim, await executor(claim))
    state = await store.read_run("run-1")
    await store.continue_tasks("run-1", "new-input", state["version"], "g1", ["r1"],
                               [{"criterion_id": "revenue", "description": "Check comparable scope"}])
    await coordinator.advance("run-1")
    state = await store.read_run("run-1")
    assert state["data"]["tasks"]["a1"]["input_version"] == 2
    await store.receive_result(old_assessment)
    await coordinator.advance("run-1")
    task = (await store.read_run("run-1"))["data"]["tasks"]["a1"]
    assert task["acceptance"] == "pending"
    assert task["state"] == "waiting_dependencies"
    assert task["accepted_ref"] is None
    assert task["context_ref"]  # Private context survives invalidation.
