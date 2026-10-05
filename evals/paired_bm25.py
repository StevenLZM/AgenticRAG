"""The real production BM25 adapter, with a narrowly guarded audit transport."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import math
import time

from agentic_rag.retrieval.adapters.elasticsearch import ElasticsearchBm25Index
from agentic_rag.retrieval.models import SearchFilter
from evals.gold_v2_models import canonical_hash
from evals.paired_retrieval_identity import comparison_arms, expected_request, validate_identity


class AuditedSearchClient:
    def __init__(self, es, ledger, *, snapshot, case, arm, identity):
        self.es, self.ledger, self.snapshot, self.case, self.arm, self.identity = es, ledger, snapshot, case, arm, identity

    async def search(self, **kwargs):
        if self.arm == "ik_max_word":
            # Check the production request before applying the one declared
            # experiment-only variation; never hide unrelated adapter drift.
            if kwargs != expected_request(self.identity, self.case.case_id, "ik"):
                raise ValueError("adapter request differs from frozen IK baseline")
            kwargs = deepcopy(kwargs)
            query = kwargs["query"]["bool"]["must"]["multi_match"]["query"]
            kwargs["query"]["bool"]["must"] = {"dis_max": {"queries": [
                {"match": {"contextualized_content.zh": {"query": query, "analyzer": "ik_max_word", "boost": 2}}},
                {"match": {"contextualized_content": query}}], "tie_breaker": 0.0}}
        if kwargs != expected_request(self.identity, self.case.case_id, self.arm):
            raise ValueError("adapter request differs from frozen DSL")
        attempt = self.ledger.start({"case_id": self.case.case_id, "arm": self.arm, "request": kwargs})
        start = time.monotonic()
        try:
            async with asyncio.timeout(self.identity["common"]["config"]["timeout_s"]):
                response = await self.es.search(**kwargs)
        except Exception as error:
            self.ledger.finish(attempt, {"status": "uncertain", "elapsed_seconds": time.monotonic() - start,
                                         "error_type": type(error).__name__})
            raise
        try:
            body = response.body
            if body.get("timed_out") or body.get("_shards", {}).get("failed") != 0:
                raise ValueError("partial_search_result")
            members = {c["child_id"]: c for c in self.snapshot.children}
            hits, seen = [], set()
            if len(body["hits"]["hits"]) > kwargs["size"]:
                raise ValueError("result_exceeds_window")
            for rank, hit in enumerate(body["hits"]["hits"], 1):
                cid, source = hit["_id"], hit["_source"]
                if cid in seen or cid not in members or source.get("id", cid) != cid:
                    raise ValueError("foreign_duplicate_hit")
                seen.add(cid)
                expected = members[cid]
                keys = ("parent_id", "user_id", "document_id", "document_version_id")
                if any(source[k] != expected[k] for k in keys):
                    raise ValueError("hit_scope_mismatch")
                if type(hit["_score"]) not in {int, float} or not math.isfinite(hit["_score"]):
                    raise ValueError("invalid_score")
                hits.append({"child_id": cid, **{k: source[k] for k in keys}, "rank": rank, "score": hit["_score"]})
            self.ledger.finish(attempt, {"status": "success", "hits": hits, "elapsed_seconds": time.monotonic() - start,
                "took_ms": body.get("took"), "response_sha256": canonical_hash(body)})
        except (KeyError, TypeError, ValueError) as error:
            self.ledger.finish(attempt, {"status": "failed", "elapsed_seconds": time.monotonic() - start,
                                         "error_type": type(error).__name__})
            raise
        return response


async def run_paired_bm25(*, es, snapshot, cases, identity, ledger):
    validate_identity(identity)
    common, config = identity["common"], identity["common"]["config"]
    expected = [{"case_id": c.case_id, "question": c.question, "sha256": canonical_hash(c.model_dump(mode="json"))} for c in cases]
    if common["cases"] != expected or snapshot.snapshot_id != common["snapshot_id"]:
        raise ValueError("runner inputs changed")
    recorded = {(r["case_id"], r["arm"]) for r in ledger.records()}
    arm_order = comparison_arms(config)
    pending = [(c, arm) for i, c in enumerate(cases) for arm in (arm_order if i % 2 == 0 else arm_order[::-1])
               if (c.case_id, arm) not in recorded]
    if not pending or ledger.remaining_wall <= 0:
        return ledger.records()
    client = es.options(request_timeout=config["timeout_s"], max_retries=0, retry_on_timeout=False)
    remaining, start = ledger.remaining_wall, time.monotonic()
    ledger.begin_phase()
    try:
        async with asyncio.timeout(remaining):
            for case, arm in pending:
                if len(ledger.records()) >= config["request_limit"]:
                    break
                proxy = AuditedSearchClient(client, ledger, snapshot=snapshot, case=case, arm=arm, identity=identity)
                adapter = ElasticsearchBm25Index(proxy, index=config["index"], index_generation=config["index_generation"],
                                                 lexical_analysis="standard" if arm == "standard" else "ik")
                try:
                    await adapter.search(case.question, SearchFilter(user_id=config["user_id"],
                        index_generation=config["index_generation"]), config["top_k"])
                except Exception:
                    # Persisted failures remain in the denominator. No retry here.
                    continue
    except TimeoutError:
        pass
    finally:
        ledger.checkpoint_elapsed(time.monotonic() - start)
    return ledger.records()
