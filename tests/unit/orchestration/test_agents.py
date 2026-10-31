import json
from importlib import import_module

import pytest
from pydantic import ValidationError

from agentic_rag.orchestration.contracts import Criterion, TaskSpec
from agentic_rag.orchestration.evidence import EvidencePreparer
from tests.unit.orchestration.test_contracts import task_payload
from tests.unit.orchestration.test_evidence import Hydrator, SCOPE, SNAPSHOT, candidate, ranked


class Execution:
    scope = SCOPE
    snapshot = SNAPSHOT

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.debits = []

    async def structured(self, role, payload, schema, *, light=False):
        self.calls.append((role, payload))
        return schema.model_validate(self.responses.pop(0))

    async def debit(self, kind, amount=1):
        self.debits.append((kind, amount))


class Retrieval:
    def __init__(self):
        self.requests = []

    async def retrieve_candidates(self, request, scope, snapshot):
        self.requests.append(request)
        return ranked().model_copy(update={"query": request.query, "document_ids": request.document_ids})


def test_analysis_output_budget_counts_executor_generated_findings():
    from agentic_rag.orchestration.agents import analysis_schema
    raw = {"findings": [{"statement": "user data", "evidence_refs": ["query"]}],
           "calculations": [{"id": "growth", "operation": "growth_percent", "criterion_ids": ["growth"],
                             "operands": [{"value": 100, "source_kind": "user_input", "evidence_ref": "query",
                                           "bundle_hash": "hash", "quote": "100", "unit": "number", "entity": "user_input", "period": "unspecified"}]}]}
    with pytest.raises(ValidationError, match="findings plus calculations"):
        analysis_schema(1).model_validate(raw)
    assert analysis_schema(2).model_validate(raw).findings


async def test_retrieval_context_resumes_without_other_tasks_and_hard_source_filter():
    module = import_module("agentic_rag.orchestration.agents")
    raw = task_payload()
    raw["sources"]["document_ids"] = ["d-p1"]
    task = TaskSpec.model_validate(raw)
    execution = Execution({"queries": ["Revenue 2025 appendix"], "reason": "target missing table"}, {
        "findings": [{"statement": "Revenue is 120 million", "evidence_refs": ["c-p1"], "criterion_ids": ["revenue"]}], "gaps": [],
    })
    retrieval = Retrieval()
    result = await module.RetrievalAgent(retrieval).execute(
        task, {"queries": ["Revenue 2025"], "summary": "missing table"}, execution, round_id=2,
    )
    assert result.round_id == 2
    assert result.context["queries"] == ["Revenue 2025", "Revenue 2025 appendix"]
    assert retrieval.requests[0].document_ids == ("d-p1",)
    assert execution.calls[0][1]["context"]["summary"] == "missing table"
    assert "other_tasks" not in json.dumps(execution.calls)
    assert execution.debits == [("queries", 1)]


async def test_retrieval_cannot_call_calculator_or_repeat_prior_query():
    module = import_module("agentic_rag.orchestration.agents")
    agent = module.RetrievalAgent(Retrieval())
    task = TaskSpec.model_validate(task_payload())
    with pytest.raises(ValidationError):
        await agent.execute(task, {}, Execution({"queries": ["revenue"], "reason": "x", "calculate": "120/100"}))
    with pytest.raises(ValueError, match="progress|repeat"):
        await agent.execute(task, {"queries": ["REVENUE"]}, Execution({"queries": ["revenue"], "reason": "x"}))


async def test_retrieval_reserves_queries_for_budgeted_gap_rounds():
    module = import_module("agentic_rag.orchestration.agents")
    raw = task_payload()
    raw["budget"] = {"max_queries": 3, "max_rounds": 3}
    raw["sources"]["document_ids"] = []
    task = TaskSpec.model_validate(raw)
    retrieval = Retrieval()
    execution = Execution({"queries": ["year 1", "year 2", "appendix"], "reason": "everything"}, {"findings": [], "gaps": []})
    with pytest.raises(ValidationError):
        await module.RetrievalAgent(retrieval).execute(task, {}, execution)
    assert retrieval.requests == []
    assert execution.calls[0][1]["query_budget"] == {"remaining_queries": 3, "remaining_rounds": 3, "max_queries_this_round": 1}


async def test_retrieval_rejects_fabricated_candidate_reference():
    module = import_module("agentic_rag.orchestration.agents")
    with pytest.raises(ValueError, match="source"):
        await module.RetrievalAgent(Retrieval()).execute(TaskSpec.model_validate(task_payload()), {}, Execution(
            {"queries": ["revenue"], "reason": "x"},
            {"findings": [{"statement": "claim", "evidence_refs": ["foreign"]}], "gaps": []},
        ))


async def inputs():
    bundle = await EvidencePreparer(Hydrator()).prepare(
        [candidate("r1", "revenue", ranked(content="ACME revenue in 2024 was 100 million; in 2025 was 120 million."))],
        [Criterion(id="revenue", description="ACME revenues")], SCOPE, SNAPSHOT,
    )
    module = import_module("agentic_rag.orchestration.agents")
    ref = next(iter(bundle.packed.manifest))
    value = module.EvidenceInput.model_validate({
        "result_ref": "grade-1", "bundle": bundle.model_dump(mode="json"),
        "grade": {"bundle_hash": bundle.bundle_hash, "decision": "sufficient", "coverage": [
            {"criterion_id": "revenue", "status": "supported", "evidence_refs": [ref]},
        ]},
    })
    return value, ref


async def test_analysis_numeric_sources_include_authorized_heading_metadata():
    module = import_module("agentic_rag.orchestration.agents")
    value, ref = await inputs()
    item = value.bundle.packed.items[0].model_copy(update={"heading_path": ("2037 annual report",)})
    packed = value.bundle.packed.model_copy(update={"items": (item,)})
    value = value.model_copy(update={"bundle": value.bundle.model_copy(update={"packed": packed})})
    finding = module.Finding(statement="The 2037 report contains revenue figures.", evidence_refs=(ref,))
    assert module.analysis_unknown_numbers([finding], [], [value], "") == []


def analysis_task():
    raw = task_payload("a1")
    raw.update(agent_type="analysis", output={"schema_name": "analysis.v1"})
    raw["tools"]["allowed"] = ["read_evidence", "calculate"]
    raw["sources"]["document_ids"] = ["d-p1"]
    return TaskSpec.model_validate(raw)


async def test_analysis_computes_only_from_cited_operands_and_returns_provenance():
    module = import_module("agentic_rag.orchestration.agents")
    value, ref = await inputs()
    quote = value.bundle.packed.items[0].content
    operands = [{"value": n, "evidence_ref": ref, "bundle_hash": value.bundle.bundle_hash,
                 "quote": quote, "unit": "million", "entity": "ACME", "period": str(year)}
                for n, year in [(100, 2024), (120, 2025)]]
    draft = {"calculations": [{"id": "growth", "operation": "growth_percent", "operands": operands,
                               "criterion_ids": ["revenue"]}], "findings": [], "gaps": []}
    result = await module.AnalysisAgent().execute(analysis_task(), [value], {}, Execution(draft), dependency_results={"r1": "grade-1"})
    assert result.status == "completed"
    assert result.calculations[0].value == 20
    assert result.calculations[0].operands[1].evidence_ref == ref
    assert result.dependency_results == {"r1": "grade-1"}
    operands[1]["value"] = 999
    bad = await module.AnalysisAgent().execute(analysis_task(), [value], {}, Execution(draft), dependency_results={"r1": "grade-1"})
    assert bad.status == "needs_input" and not bad.calculations
    assert bad.gaps


@pytest.mark.parametrize("invalid", ["source", "number"])
async def test_analysis_repairs_unbound_sources_or_unaudited_arithmetic(invalid):
    from copy import deepcopy
    from agentic_rag.orchestration.agents import AnalysisAgent
    value, ref = await inputs()
    quote = value.bundle.packed.items[0].content
    raw = {"calculations": [{"id": "growth", "operation": "growth_percent", "criterion_ids": ["revenue"],
        "operands": [{"value": n, "evidence_ref": ref, "bundle_hash": value.bundle.bundle_hash, "quote": quote,
                      "unit": "million", "entity": "ACME", "period": str(year)} for n, year in ((100, 2024), (120, 2025))]}],
        "findings": [{"statement": "Revenue increased", "kind": "inference", "evidence_refs": ["query"],
                      "criterion_ids": ["revenue"], "calculation_refs": ["growth"]}]}
    corrected = deepcopy(raw)
    corrected["findings"][0]["evidence_refs"] = [ref]
    if invalid == "number":
        raw["findings"][0]["evidence_refs"] = [ref]
        raw["findings"][0]["statement"] = "Revenue grew 20%, a difference of 10 percentage points"
        corrected["findings"][0]["statement"] = "Revenue grew 20%"

    class RepairExecution(Execution):
        async def structured(self, role, payload, schema, *, light=False):
            with pytest.raises(ValidationError, match="finding source" if invalid == "source" else "unsupported numeric"):
                schema.model_validate(raw)
            return schema.model_validate(corrected)

    result = await AnalysisAgent().execute(analysis_task(), [value], {}, RepairExecution(), dependency_results={"r1": "grade-1"})
    assert result.calculations[0].value == 20
    assert all("query" not in finding.evidence_refs for finding in result.findings)


async def test_schema_validation_never_calculates_when_task_withholds_calculate(monkeypatch):
    module = import_module("agentic_rag.orchestration.agents")
    value, ref = await inputs()
    task = analysis_task()
    task = task.model_copy(update={"tools": task.tools.model_copy(update={"allowed": ("read_evidence",)})})
    raw = {"calculations": [{"id": "sum", "operation": "sum", "criterion_ids": ["revenue"],
        "operands": [{"value": 100, "evidence_ref": ref, "bundle_hash": value.bundle.bundle_hash,
                      "quote": value.bundle.packed.items[0].content, "unit": "million", "entity": "ACME", "period": "2024"}]}]}
    def forbidden(*args, **kwargs):
        raise AssertionError("calculator invoked without task permission")
    monkeypatch.setattr(module, "calculate", forbidden)
    with pytest.raises(ValidationError, match="authorize calculation"):
        await module.AnalysisAgent().execute(task, [value], {}, Execution(raw), dependency_results={"r1": "grade-1"})


async def test_grader_rejects_a_supported_claim_without_manifest_evidence():
    module = import_module("agentic_rag.orchestration.agents")
    value, _ = await inputs()
    execution = Execution({"bundle_hash": value.bundle.bundle_hash, "decision": "sufficient", "coverage": [
        {"criterion_id": "revenue", "status": "supported", "evidence_refs": ["not-in-bundle"]},
    ]})
    with pytest.raises(ValueError, match="evidence"):
        await module.GroupGrader().grade(value.bundle, execution)


async def test_user_arithmetic_needs_only_calculate_but_external_evidence_requires_read_permission():
    module = import_module("agentic_rag.orchestration.agents")
    from agentic_rag.orchestration.contracts import content_hash
    task = TaskSpec.model_validate({
        **analysis_task().model_dump(), "tools": {"allowed": ["calculate"], "instructions": "User supplied operands only"},
    })
    question = "Revenue grows from 100 to 120. Compute growth."
    execution = Execution({"findings": [{"statement": "Growth compares the two user-provided revenues.", "kind": "inference", "evidence_refs": ["query"], "criterion_ids": ["revenue"]}], "calculations": [{"id": "growth", "operation": "growth_percent", "criterion_ids": ["revenue"],
        "operands": [{"source_kind": "user_input", "value": value, "evidence_ref": "query", "bundle_hash": content_hash({"question": question}),
                      "quote": question, "unit": "number", "entity": "user_input", "period": "unspecified"} for value in (100, 120)]}], "gaps": []})
    execution.question = question
    result = await module.AnalysisAgent().execute(task, [], {}, execution, dependency_results={})
    assert result.calculations[0].value == 20
    value, _ = await inputs()
    with pytest.raises(ValueError, match="read_evidence"):
        await module.AnalysisAgent().execute(task, [value], {}, Execution(), dependency_results={"r1": "grade-1"})


def test_query_operand_requires_explicit_user_input_binding():
    from agentic_rag.orchestration.agents import Operand
    with pytest.raises(ValidationError, match="user_input"):
        Operand(value=100, evidence_ref="query", bundle_hash="hash", quote="100", unit="number", entity="user_input", period="unspecified")


def test_user_operand_cannot_add_unprovided_entity_metadata():
    from agentic_rag.orchestration.agents import CalculationRequest, calculate
    from agentic_rag.orchestration.contracts import content_hash
    question = "Acme revenue was 100 million in 2024."
    request = CalculationRequest(id="sum", operation="sum", criterion_ids=("revenue",), operands=({
        "source_kind": "user_input", "value": 100, "evidence_ref": "query", "bundle_hash": content_hash({"question": question}),
        "quote": question, "entity": "Beijing Acme", "unit": "million", "period": "2024",
    },))
    with pytest.raises(ValueError, match="metadata"):
        calculate(request, [], question=question)


@pytest.mark.parametrize("also_exceeds_item_budget", [False, True])
async def test_user_operand_metadata_error_enters_reserved_schema_repair(store, also_exceeds_item_budget):
    from copy import deepcopy
    from types import SimpleNamespace
    from agentic_rag.orchestration.agents import AnalysisAgent
    from agentic_rag.orchestration.contracts import content_hash
    from agentic_rag.orchestration.execution import ModelExecution
    from agentic_rag.runtime.model_gateway import ModelGateway
    question = "凌星科技2024年收入为100亿元人民币，2025年收入为120亿元人民币。"
    raw = {"calculations": [{"id": "growth", "operation": "growth_percent", "criterion_ids": ["revenue"],
        "operands": [{"source_kind": "user_input", "value": n, "evidence_ref": "query", "bundle_hash": content_hash({"question": question}),
                      "quote": question, "unit": "人民币亿元", "entity": "凌星科技", "period": str(year)} for n, year in ((100, 2024), (120, 2025))]}]}
    repaired = deepcopy(raw)
    for operand in repaired["calculations"][0]["operands"]:
        operand["unit"] = "亿元人民币"
    task = analysis_task()
    if also_exceeds_item_budget:
        raw["findings"] = [{"statement": "Revenue comparison", "evidence_refs": ["query"]}]
        raw["calculations"][0]["operands"][0]["entity"] = "未提供的公司"
        task = task.model_copy(update={"output": task.output.model_copy(update={"max_items": 1})})

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
            self.requests = []
        async def create(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 2:
                feedback = kwargs["messages"][-1]["content"].split("Validation error:", 1)[1]
                assert "operand metadata unit" in feedback
                if also_exceeds_item_budget:
                    assert "findings plus calculations" in feedback
                    assert "metadata entity" in feedback
            return {"model": "test", "choices": [{"message": {"content": json.dumps(raw if len(self.requests) == 1 else repaired)}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 500}}

    from tests.unit.orchestration.test_store import dispatch
    claim = await store.claim((await dispatch(store))[0], "analysis-repair")
    client = Client()
    execution = ModelExecution(store, claim, ModelGateway(client, max_retries=0))
    execution.question = question
    result = await AnalysisAgent().execute(task, [], {}, execution, dependency_results={})
    assert len(client.requests) == 2
    assert result.calculations[0].value == 20
    assert result.calculations[0].operands[0].unit == "亿元人民币"
    state = (await store.read_run("run-1"))["data"]
    assert state["budget"]["model_calls"]["used"] == 2
    assert state["budget"]["output_tokens"]["used"] == 1000
