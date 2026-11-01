"""Offline A/B/C/D/E analysis of recorded runs; never dispatches a model."""

import math
import re
from collections import defaultdict
from statistics import mean
from pydantic import BaseModel, ConfigDict, Field, model_validator
from agentic_rag.models.contextual import SHA256_PATTERN

ABLATIONS = {
    "A": "legacy 384 split / original rerank / original query",
    "B": "320 split / original rerank / original query",
    "C": "same 320 split / structural context in recall and rerank",
    "D": "same split / grounded independently verified LLM context",
    "E": "same D index / policy-compiled query",
}


class SourceSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    document_id: str
    version_id: str
    block: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("source range must be nonempty")
        return self


def contains_gold(candidates, gold):
    spans = [
        SourceSpan.model_validate(value)
        for candidate in candidates
        for value in candidate["spans"]
    ]
    return all(
        any(
            (hit.document_id, hit.version_id, hit.block)
            == (wanted.document_id, wanted.version_id, wanted.block)
            and hit.start <= wanted.start
            and hit.end >= wanted.end
            for hit in spans
        )
        for wanted in gold
    )


def source_metrics(candidates, gold, *, required_support=(), k=10):
    if k < 1 or not gold:
        raise ValueError("positive K and source gold required")
    relevant = [contains_gold([candidate], gold) for candidate in candidates[:k]]
    rank = next((index + 1 for index, found in enumerate(relevant) if found), None)
    gains = [1 / math.log2(index + 2) for index, found in enumerate(relevant) if found]
    # A gold fact is one binary target; duplicate overlapping children add no
    # further relevance credit beyond the first occurrence.
    return {
        "recall": float(contains_gold(candidates[:k], gold)),
        "reciprocal_rank": 1 / rank if rank else 0.0,
        "ndcg": gains[0] if gains else 0.0,
        "source_support_failures": sum(
            not contains_gold(candidates[:k], [source]) for source in required_support
        ),
    }


def validate_ablation_configs(configs):
    if set(configs) != set(ABLATIONS):
        raise ValueError("all A/B/C/D/E configurations required")
    fixed = (
        "corpus_hash",
        "acl_hash",
        "embedding",
        "reranker",
        "recall_k",
        "rrf_k",
        "rerank_k",
        "task_budget",
    )
    if any(
        any(config[key] != configs["A"][key] for key in fixed)
        for config in configs.values()
    ):
        raise ValueError(
            "ablation requires equal source corpus, models, ACL and candidate/task budgets"
        )
    if configs["A"]["base_tokens"] != 384 or any(
        configs[key]["base_tokens"] != 320 for key in "BCDE"
    ):
        raise ValueError("measure 384 to 320 split independently in A/B")
    if len({configs[key]["source_boundaries_hash"] for key in "BCDE"}) != 1:
        raise ValueError("B/C/D/E require identical original source boundaries")
    identities = [configs[key].get("index_manifest_hash") for key in "DE"]
    if any(
        not isinstance(identity, str) or not re.fullmatch(SHA256_PATTERN, identity)
        for identity in identities
    ):
        raise ValueError("D/E require explicit SHA-256 index manifest identities")
    if identities[0] != identities[1]:
        raise ValueError("D/E require exactly the same index build")


def summarize_recorded(records):
    grouped = defaultdict(list)
    for row in records:
        if row["config"] not in ABLATIONS or not row["run_id"] or not row["build"]:
            raise ValueError("run/config/build provenance required")
        for key in (
            "recall",
            "reciprocal_rank",
            "ndcg",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "cost",
            "attribution_failures",
            "source_support_failures",
        ):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("recorded metrics must be finite and nonnegative")
        grouped[row["config"]].append(row)
    result = {"evidence_kind": "recorded_runs_not_fresh_model_quality"}
    for config, rows in grouped.items():
        latency = sorted(row["latency_ms"] for row in rows)
        result[config] = {
            "count": len(rows),
            "runs": [(r["run_id"], r["build"]) for r in rows],
            **{
                key: mean(row[key] for row in rows)
                for key in ("recall", "reciprocal_rank", "ndcg")
            },
            **{
                key: sum(row[key] for row in rows)
                for key in (
                    "attribution_failures",
                    "source_support_failures",
                    "input_tokens",
                    "output_tokens",
                    "cost",
                )
            },
            "p95_latency_ms": latency[math.ceil(0.95 * len(latency)) - 1],
        }
    return result
