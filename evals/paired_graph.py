"""Read-only experiment boundaries around the unmodified production QueryGraph."""
from __future__ import annotations

from copy import deepcopy
import time

from evals.gold_rebinding import _json, _write
from evals.gold_v2_models import canonical_hash
from evals.retrieval_metrics_v2 import score_context


class GraphSearchClient:
    """Change only the Chinese query analyzer; persist actual outbound searches."""

    def __init__(self, es, *, index, user_id, generation, children=None):
        self.es, self.index, self.user_id, self.generation = es, index, user_id, generation
        self.children = None if children is None else {c["child_id"]: c for c in children}
        self.records = []
        self.arm = self.path = None

    def begin(self, arm, audit_path):
        if arm not in {"ik", "ik_max_word"} or audit_path.exists():
            raise ValueError("invalid arm or existing search journal")
        self.arm, self.path, self.records = arm, audit_path, []

    async def search(self, **kwargs):
        if self.path is None or kwargs.get("index") != self.index:
            raise ValueError("unbound or foreign index")
        request = deepcopy(kwargs)
        dense = "knn" in request
        filters = request["knn"]["filter"] if dense else request["query"]["bool"]["filter"]
        expected = {"user_id": self.user_id, "is_active": True, "index_generation": self.generation}
        for key, value in expected.items():
            terms = [f["term"][key] for f in filters if key in f.get("term", {})]
            if terms != [value]:
                raise ValueError("search scope mismatch")
        if not dense:
            text = request["query"]["bool"]["must"]
            query = text.get("multi_match", {}).get("query")
            if not isinstance(query, str) or text != {"multi_match": {
                "query": query, "fields": ["contextualized_content.zh^2", "contextualized_content"],
                "type": "best_fields", "tie_breaker": 0.0,
            }}:
                raise ValueError("production lexical profile changed")
            if self.arm == "ik_max_word":
                request["query"]["bool"]["must"] = {"dis_max": {"queries": [
                    {"match": {"contextualized_content.zh": {"query": query, "analyzer": "ik_max_word", "boost": 2}}},
                    {"match": {"contextualized_content": query}}], "tie_breaker": 0.0}}
        audit = deepcopy(request)
        if dense:
            vector = audit["knn"].pop("query_vector")
            audit["knn"].update(vector_sha256=canonical_hash(vector), vector_dimensions=len(vector))
        record = {**audit, "ordinal": len(self.records), "arm": self.arm, "status": "running"}
        self.records.append(record)
        _write(self.path, self.records)  # Persist before transport, including concurrent lanes.
        start = time.monotonic()
        try:
            response = await self.es.search(**request)
            body = response.body
            if body.get("timed_out") or body.get("_shards", {}).get("failed") != 0:
                raise ValueError("partial search result")
            hits = body["hits"]["hits"]
            if len(hits) > request["size"] or len({h["_id"] for h in hits}) != len(hits):
                raise ValueError("invalid search result window")
            for hit in hits:
                if hit["_source"]["user_id"] != self.user_id:
                    raise ValueError("foreign search hit")
                if self.children is not None:
                    source, member = hit["_source"], self.children.get(hit["_id"])
                    if member is None or any(source[k] != member[k] for k in
                            ("parent_id", "document_id", "document_version_id", "user_id")):
                        raise ValueError("hit outside frozen corpus")
            record.update(status="success", hits=[{"child_id": h["_id"], "score": h["_score"]} for h in hits],
                          response_sha256=canonical_hash(body))
            return response
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - start
            _write(self.path, self.records)


async def query_once(path, binding, operation):
    """At-most-once query dispatch. A crash is uncertain, never permission to resend."""
    if path.exists():
        record = _json(path)
        if record.get("binding") != binding or record.get("status") not in {"running", "completed", "failed"}:
            raise ValueError("query journal binding/status mismatch")
        if record["status"] == "completed" and record.get("result_sha256") != canonical_hash(record["result"]):
            raise ValueError("query result hash mismatch")
        return record
    record = {"binding": binding, "status": "running"}
    _write(path, record)
    try:
        result = await operation()
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__)
    else:
        record.update(status="completed", result=result, result_sha256=canonical_hash(result))
    _write(path, record)
    return record


def select_graph_cases(source_identity, cases):
    by_id = {c.case_id: c for c in cases}
    selected = []
    for entry in source_identity["common"]["cases"]:
        case = by_id.get(entry["case_id"])
        if case is None or entry != {"case_id": case.case_id, "question": case.question,
                                     "sha256": canonical_hash(case.model_dump(mode="json"))}:
            raise ValueError("frozen gold changed")
        if case.category == "multi_hop":
            selected.append(case)
    if len(selected) != 20 or len({c.case_id for c in selected}) != 20:
        raise ValueError("expected original twenty multi-hop cases")
    return selected


class GraderObserver:
    """Observe the real grader without changing prompts, evidence or return values."""

    def __init__(self, grader, path):
        self.grader, self.path, self.records = grader, path, []

    async def grade(self, question, packed, **kwargs):
        record = {"question": question, "items": [i.model_dump(mode="json") for i in packed.items],
                  "status": "running"}
        self.records.append(record)
        _write(self.path, self.records)
        try:
            grade = await self.grader.grade(question, packed, **kwargs)
            record.update(status="completed", grade=grade.model_dump(mode="json"))
            return grade
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            _write(self.path, self.records)


def grade_diagnostics(case, records, final_items):
    history, missing_seen, recovered = [], False, False
    for record in records:
        coverage = score_context(case, record["items"])
        decision = record.get("grade", {}).get("decision")
        complete = coverage["complete_evidence"] == 1.0
        if not complete and decision == "insufficient":
            missing_seen = True
        if missing_seen and complete and decision == "sufficient":
            recovered = True
        history.append({"decision": decision, "gaps": record.get("grade", {}).get("gaps", []), **coverage})
    final = score_context(case, final_items)
    return {"history": history, "gap_recovery": recovered and final["complete_evidence"] == 1.0,
            "false_sufficient_count": sum(h["decision"] == "sufficient" and not h["complete_evidence"] for h in history),
            "final_complete_evidence": final["complete_evidence"], "final_fact_recall": final["fact_recall"]}
