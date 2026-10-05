from copy import deepcopy

import pytest

from agentic_rag.retrieval.adapters.elasticsearch import ElasticsearchBm25Index, ElasticsearchVectorIndex
from agentic_rag.retrieval.models import SearchFilter
from evals.gold_v2_models import GoldCaseV2, canonical_hash
from tests.unit.evals.test_gold_v2_validation import gold_payload
from tests.unit.evals.test_paired_bm25 import SearchES


async def test_graph_transport_changes_only_chinese_bm25_clause(tmp_path):
    from evals.paired_graph import GraphSearchClient
    raw = SearchES()
    client = GraphSearchClient(raw, index="children", user_id="u", generation="index")
    client.begin("ik_max_word", tmp_path / "search.json")
    scope = SearchFilter(user_id="u", index_generation="index")
    await ElasticsearchBm25Index(client, index="children", index_generation="index", lexical_analysis="ik").search("2025年", scope, 40)
    await ElasticsearchVectorIndex(client, index="children", index_generation="index").search([0.1] * 1024, scope, 40)
    query = raw.requests[0]["query"]["bool"]
    assert query["must"] == {"dis_max": {"queries": [
        {"match": {"contextualized_content.zh": {"query": "2025年", "analyzer": "ik_max_word", "boost": 2}}},
        {"match": {"contextualized_content": "2025年"}}], "tie_breaker": 0.0}}
    assert query["filter"] == [{"term": {"user_id": "u"}}, {"term": {"is_active": True}}, {"term": {"index_generation": "index"}}]
    assert raw.requests[1]["knn"]["query_vector"] == [0.1] * 1024
    assert raw.requests[1]["size"] == 40
    assert client.records[0]["query"] == raw.requests[0]["query"]
    assert "query_vector" not in str(client.records)  # Do not duplicate dense vectors in the audit.


@pytest.mark.parametrize("fault", ["scope", "profile", "index"])
async def test_graph_transport_rejects_scope_and_adapter_drift_before_send(tmp_path, fault):
    from evals.paired_graph import GraphSearchClient
    from evals.paired_retrieval_identity import expected_request
    from tests.unit.evals.test_paired_retrieval_identity import identity_fixture
    case, identity = identity_fixture()
    request = expected_request(identity, case.case_id, "ik")
    client = GraphSearchClient(SearchES(), index=request["index"], user_id="u", generation=request["query"]["bool"]["filter"][2]["term"]["index_generation"])
    client.begin("ik_max_word", tmp_path / "search.json")
    if fault == "scope":
        request["query"]["bool"]["filter"][0] = {"term": {"user_id": "foreign"}}
    elif fault == "profile":
        request["query"]["bool"]["must"]["multi_match"]["fields"] = ["content"]
    else:
        request["index"] = "foreign"
    with pytest.raises(ValueError):
        await client.search(**request)
    assert client.records == []


async def test_paid_query_journal_resumes_without_executing_again(tmp_path):
    from evals.paired_graph import query_once
    calls = []
    async def paid():
        calls.append(1)
        return {"answer": "actual answer"}
    path = tmp_path / "case.json"
    first = await query_once(path, "binding", paid)
    before = path.read_bytes()
    assert await query_once(path, "binding", paid) == first
    assert calls == [1]
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        await query_once(path, "different", paid)


async def test_interrupted_query_is_uncertain_not_retried(tmp_path):
    from evals.paired_graph import query_once
    async def crashed():
        raise KeyboardInterrupt()
    path = tmp_path / "case.json"
    with pytest.raises(KeyboardInterrupt):
        await query_once(path, "binding", crashed)
    async def forbidden():
        pytest.fail("uncertain query was sent twice")
    assert (await query_once(path, "binding", forbidden))["status"] == "running"


async def test_completed_query_tampering_is_rejected(tmp_path):
    from evals.paired_graph import query_once
    from evals.gold_rebinding import _json, _write
    async def paid():
        return {"answer": "actual"}
    path = tmp_path / "case.json"
    await query_once(path, "binding", paid)
    value = _json(path)
    value["result"]["answer"] = "changed"
    _write(path, value)
    with pytest.raises(ValueError, match="hash"):
        await query_once(path, "binding", paid)


def test_case_selection_preserves_original_twenty_and_rejects_changed_gold():
    from evals.paired_graph import select_graph_cases
    cases = [GoldCaseV2.model_validate({**gold_payload(), "case_id": f"m{i}"}) for i in range(20)]
    source = {"common": {"cases": [{"case_id": c.case_id, "question": c.question,
        "sha256": canonical_hash(c.model_dump(mode="json"))} for c in cases]}}
    assert [c.case_id for c in select_graph_cases(source, cases[::-1])] == [f"m{i}" for i in range(20)]
    changed = [cases[0].model_copy(update={"question": "new question"}), *cases[1:]]
    with pytest.raises(ValueError):
        select_graph_cases(source, changed)


def test_grade_diagnostic_measures_packed_facts_not_just_retrieval_ids():
    from evals.paired_graph import grade_diagnostics
    case = GoldCaseV2.model_validate(gold_payload())
    missing = {"parent_id": "p", "content": "410", "heading_path": []}
    full = {**missing, "content": "410 470"}
    records = [{"grade": {"decision": "insufficient", "gaps": ["missing new limit"]}, "items": [missing]},
               {"grade": {"decision": "sufficient", "gaps": []}, "items": [full]}]
    result = grade_diagnostics(case, records, [full])
    assert result["gap_recovery"] is True
    assert result["false_sufficient_count"] == 0
    assert result["final_complete_evidence"] == 1.0
    misleading = deepcopy(records)
    misleading[1]["items"] = [missing]
    result = grade_diagnostics(case, misleading, [missing])
    assert result["gap_recovery"] is False
    assert result["false_sufficient_count"] == 1


async def test_grader_observer_returns_real_grade_without_gold_injection(tmp_path):
    from evals.paired_graph import GraderObserver
    from agentic_rag.models.schemas import EvidenceGradeV2
    from agentic_rag.query.evidence_builder import PackedEvidence
    original = EvidenceGradeV2(decision="insufficient", gap_type="missing_facts", gaps=("missing year",))
    class Grader:
        async def grade(self, question, packed, **kwargs):
            assert question == "real question" and set(kwargs) == {"scope"}
            return original
    observer = GraderObserver(Grader(), tmp_path / "grades.json")
    packed = PackedEvidence(items=(), manifest={}, rendered_context="", token_count=0, index_generation="index")
    assert await observer.grade("real question", packed, scope="u") is original
    assert observer.records[0]["grade"]["gaps"] == ["missing year"]
