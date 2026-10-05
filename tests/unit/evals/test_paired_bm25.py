from copy import deepcopy
from types import SimpleNamespace

import pytest

from evals.gold_v2_models import CorpusSnapshot
from tests.unit.evals.test_gold_v2_validation import snapshot_payload
from tests.unit.evals.test_paired_retrieval_identity import identity_fixture


class SearchES:
    def __init__(self, fault=None):
        self.requests, self.configs, self.fault = [], [], fault

    def options(self, **kwargs):
        self.configs.append(kwargs)
        return self

    async def search(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if self.fault == "timeout":
            raise TimeoutError()
        hits = [{"_id": cid, "_score": 1.0, "_source": {"id": cid, "parent_id": "p", "document_id": "d",
            "document_version_id": "v", "user_id": "u", "content": "fact", "ast_locator": ""}} for cid in ("c2", "c1")]
        if self.fault == "foreign":
            hits[0]["_source"]["user_id"] = "other"
        if self.fault == "duplicate":
            hits[1] = deepcopy(hits[0])
        return SimpleNamespace(body={"hits": {"hits": hits}, "timed_out": False, "_shards": {"failed": 0}, "took": 1})


async def test_adapter_kwargs_equal_saved_request_and_native_tied_order(tmp_path):
    from evals.paired_bm25 import run_paired_bm25
    from evals.paired_retrieval_ledger import PairedLedger
    case, identity = identity_fixture()
    es = SearchES()
    with PairedLedger.open(tmp_path / "run", identity, resume=False) as ledger:
        records = await run_paired_bm25(es=es, snapshot=CorpusSnapshot.model_validate(snapshot_payload()),
                                      cases=[case], identity=identity, ledger=ledger)
    assert len(es.requests) == 2
    assert [r["request"] for r in records] == es.requests
    assert [h["child_id"] for h in records[0]["hits"]] == ["c2", "c1"]
    assert es.requests[0]["query"]["bool"]["must"] == {"match": {"contextualized_content": case.question}}
    assert es.requests[1]["query"]["bool"]["must"]["multi_match"]["fields"] == ["contextualized_content.zh^2", "contextualized_content"]
    assert es.configs[0]["max_retries"] == 0 and es.configs[0]["retry_on_timeout"] is False


@pytest.mark.parametrize("fault, status", [("timeout", "uncertain"), ("foreign", "failed"), ("duplicate", "failed")])
async def test_native_hit_scope_and_failures_retained(tmp_path, fault, status):
    from evals.paired_bm25 import run_paired_bm25
    from evals.paired_retrieval_ledger import PairedLedger
    case, identity = identity_fixture()
    with PairedLedger.open(tmp_path / "run", identity, resume=False) as ledger:
        records = await run_paired_bm25(es=SearchES(fault), snapshot=CorpusSnapshot.model_validate(snapshot_payload()),
                                      cases=[case], identity=identity, ledger=ledger)
    assert [r["status"] for r in records] == [status, status]


async def test_complete_resume_adds_zero_searches(tmp_path):
    from evals.paired_bm25 import run_paired_bm25
    from evals.paired_retrieval_ledger import PairedLedger
    case, identity = identity_fixture()
    es, root = SearchES(), tmp_path / "run"
    kwargs = dict(es=es, snapshot=CorpusSnapshot.model_validate(snapshot_payload()), cases=[case], identity=identity)
    with PairedLedger.open(root, identity, resume=False) as ledger:
        before = await run_paired_bm25(**kwargs, ledger=ledger)
    ledger_bytes = (root / "ledger.json").read_bytes()
    with PairedLedger.open(root, identity, resume=True) as ledger:
        assert await run_paired_bm25(**kwargs, ledger=ledger) == before
    assert len(es.requests) == 2
    assert (root / "ledger.json").read_bytes() == ledger_bytes


async def test_analyzer_ablation_audits_actual_dsl_alternates_and_resumes_without_search(tmp_path):
    from evals.paired_bm25 import run_paired_bm25
    from evals.paired_retrieval_ledger import PairedLedger
    case, _ = identity_fixture()
    second = case.model_copy(update={"case_id": "second", "question": "2026年住宿100元限3天"})
    _, identity = identity_fixture(comparison="ik-smart-vs-max", extra_cases=[second])
    es, root = SearchES(), tmp_path / "run"
    kwargs = dict(es=es, snapshot=CorpusSnapshot.model_validate(snapshot_payload()), cases=[case, second], identity=identity)
    with PairedLedger.open(root, identity, resume=False) as ledger:
        records = await run_paired_bm25(**kwargs, ledger=ledger)
    assert [(r["case_id"], r["arm"]) for r in records] == [
        (case.case_id, "ik"), (case.case_id, "ik_max_word"), ("second", "ik_max_word"), ("second", "ik")]
    assert all(r["status"] == "success" for r in records)
    assert [r["request"] for r in records] == es.requests
    assert es.requests[1]["query"]["bool"]["must"] == {"dis_max": {"queries": [
        {"match": {"contextualized_content.zh": {"query": case.question, "analyzer": "ik_max_word", "boost": 2}}},
        {"match": {"contextualized_content": case.question}}], "tie_breaker": 0.0}}
    ledger_bytes = (root / "ledger.json").read_bytes()
    with PairedLedger.open(root, identity, resume=True) as ledger:
        assert await run_paired_bm25(**kwargs, ledger=ledger) == records
    assert len(es.requests) == 4 and (root / "ledger.json").read_bytes() == ledger_bytes


async def test_analyzer_override_does_not_mask_production_adapter_drift(tmp_path):
    from evals.paired_bm25 import AuditedSearchClient
    from evals.paired_retrieval_identity import expected_request
    from evals.paired_retrieval_ledger import PairedLedger
    case, identity = identity_fixture(comparison="ik-smart-vs-max")
    request = expected_request(identity, case.case_id, "ik")
    request["query"]["bool"]["must"]["multi_match"]["fields"][0] = "contextualized_content.zh^3"
    es = SearchES()
    with PairedLedger.open(tmp_path / "run", identity, resume=False) as ledger:
        client = AuditedSearchClient(es, ledger, snapshot=CorpusSnapshot.model_validate(snapshot_payload()),
                                    case=case, arm="ik_max_word", identity=identity)
        with pytest.raises(ValueError, match="adapter request"):
            await client.search(**request)
        assert ledger.records() == [] and es.requests == []
