from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json

from evals.routing import load_cases, RoutingSample, run_routing_eval, score_routing, dataset_hash
from agentic_rag.models.schemas import RouteDecision
from tests.unit.query.test_router_fast_path import _initial_state
from agentic_rag.runtime.models import RuntimeConfigSnapshot


DATASET = Path("evals/datasets/routing_v2.jsonl")


def perfect(cases):
    return [RoutingSample(case_id=c.id, variant=v, repeat=r, actual_route=c.expected_route,
        response_mode=c.expected_response_mode, normalized_query=c.question + " ".join(c.normalized_query_must_include),
        latency_ms=1, model_id="test", prompt_hash="a" * 64, dataset_sha256=dataset_hash(cases))
        for c in cases for r in range(3) for v in ("v1", "v2")]


def test_dataset_frozen_shape():
    cases = load_cases(DATASET)
    assert len(cases) == len({c.id for c in cases}) == 60
    assert sorted(Counter(c.group for c in cases).values()) == [10, 10, 10, 10, 20]
    assert sum(c.core for c in cases) == 8


def test_complete_incomplete_and_duplicate_samples():
    cases = load_cases(DATASET)
    samples = perfect(cases)
    assert score_routing(samples, cases)["status"] == "PASS"
    assert score_routing(samples[:-1], cases)["status"] == "INCOMPLETE"
    assert score_routing(samples + [samples[-1]], cases)["status"] == "INCOMPLETE"
    samples[-1] = samples[-1].model_copy(update={"error_code": "provider_unavailable"})
    assert score_routing(samples, cases)["status"] == "INCOMPLETE"


def test_all_clarify_is_not_success_and_entities_are_checked():
    cases = load_cases(DATASET)
    samples = perfect(cases)
    wrong = [s.model_copy(update={"actual_route": "chat", "response_mode": "clarify"}) for s in samples]
    report = score_routing(wrong, cases)
    assert report["status"] == "FAIL"
    assert report["variants"]["v2"]["missed_retrieval_rate"] == 1
    assert report["variants"]["v2"]["clarification_rate"] == 1
    core = next(c for c in cases if c.core and c.normalized_query_must_include)
    corrupted = [s.model_copy(update={"normalized_query": "做了几年"}) if s.case_id == core.id else s for s in samples]
    assert score_routing(corrupted, cases)["status"] == "FAIL"


def test_exact_accuracy_and_directional_thresholds():
    cases = load_cases(DATASET)
    core_ids = {c.id for c in cases if c.core}
    samples = perfect(cases)
    candidates = [i for i, s in enumerate(samples) if s.variant == "v2" and s.case_id not in core_ids]
    for failures, status in [(9, "PASS"), (10, "FAIL")]:
        rows = samples.copy()
        for i in candidates[:failures]:
            rows[i] = rows[i].model_copy(update={"response_mode": "wrong"})
        assert score_routing(rows, cases)["status"] == status
    labels = {c.id: c for c in cases}
    for expected_chat, field in [(True, "false_retrieval"), (False, "missed_retrieval")]:
        indices = [i for i, s in enumerate(samples) if s.variant == "v2" and s.case_id not in core_ids
                   and (labels[s.case_id].expected_route == "chat") == expected_chat]
        denominator = sum((c.expected_route == "chat") == expected_chat for c in cases) * 3
        limit = int(denominator * .05)
        for count, status in [(limit, "PASS"), (limit + 1, "FAIL")]:
            rows = samples.copy()
            for i in indices[:count]:
                rows[i] = rows[i].model_copy(update={"actual_route": "fast_rag" if expected_chat else "chat"})
            report = score_routing(rows, cases)
            assert report["status"] == status
            assert report["variants"]["v2"][field + "_denominator"] == denominator


async def test_runner_only_calls_classifier_and_interleaves_variants():
    cases = load_cases(DATASET)
    calls = []
    class Gateway:
        async def complete_structured(self, call, schema):
            payload = json.loads(call.messages[1]["content"])
            case = next(c for c in cases if c.question == payload["question"])
            calls.append(schema.__name__)
            if schema is RouteDecision:
                value = schema(route=case.expected_route, normalized_query=case.question, reason_code="test")
            else:
                value = schema(required_sources=case.required_source_set,
                    retrieval_complexity=("multi" if case.expected_route == "research" else "single") if "knowledge_base" in case.required_source_set else "none",
                    needs_clarification=case.expected_response_mode == "clarify",
                    normalized_query=case.question + " ".join(case.normalized_query_must_include), reason_code="general_conversation")
            return SimpleNamespace(value=value, actual_model="test")
    snapshot = RuntimeConfigSnapshot.model_validate(_initial_state()["runtime_config_snapshot"])
    samples = await run_routing_eval(cases, Gateway(), snapshot)
    assert len(samples) == 360
    assert calls == ["RouteDecision", "RouteAssessment"] * 180
    assert score_routing(samples, cases)["status"] == "PASS"
