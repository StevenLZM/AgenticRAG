"""A separate lexical paired protocol; no relaxed formal-evaluation fingerprints."""
from __future__ import annotations

from copy import deepcopy
import hashlib

from evals.gold_v2_models import canonical_hash

PROTOCOL = "ik-bm25-paired-v1"
CATEGORIES = ("multi_hop", "section_comprehension", "single_hop", "table_calculation", "unanswerable")
PROFILES = {
    "standard": {"kind": "match", "fields": {"contextualized_content": 1}, "operator": "OR"},
    "ik": {"kind": "multi_match", "type": "best_fields", "fields": {"contextualized_content.zh": 2,
           "contextualized_content": 1}, "tie_breaker": 0.0, "operator": "OR"},
}
MAX_WORD_PROFILE = {"kind": "dis_max", "fields": {"contextualized_content.zh": 2,
    "contextualized_content": 1}, "tie_breaker": 0.0, "operator": "OR",
    "search_analyzers": {"contextualized_content.zh": "ik_max_word", "contextualized_content": "standard"}}
COMPARISONS = {"standard-vs-ik": ("standard", "ik"), "ik-smart-vs-max": ("ik", "ik_max_word")}
SOURCE_FIELDS = ["id", "parent_id", "user_id", "document_id", "document_version_id", "content", "ast_locator"]


def comparison_arms(config):
    name = config.get("comparison", "standard-vs-ik")
    if not isinstance(name, str) or name not in COMPARISONS:
        raise ValueError("unsupported paired comparison")
    return COMPARISONS[name]


def _profile(name):
    return MAX_WORD_PROFILE if name == "ik_max_word" else PROFILES.get(name)


def select_paired_cases(cases, *, per_category=20):
    if type(per_category) is not int or per_category <= 0:
        raise ValueError("invalid per-category count")
    if len({c.case_id for c in cases}) != len(cases) or any(c.category not in CATEGORIES for c in cases):
        raise ValueError("duplicate cases or unknown category")
    selected = []
    for category in CATEGORIES:
        pool = [c for c in cases if c.split == "dev" and c.category == category]
        if len(pool) < per_category:
            raise ValueError("insufficient dev category: " + category)
        selected.extend(sorted(pool, key=lambda c: (hashlib.sha256((PROTOCOL + ":" + c.case_id).encode()).hexdigest(), c.case_id))[:per_category])
    return selected


def build_paired_identity(*, bundle_audit, environment, cases, config, implementation):
    arm_order = comparison_arms(config)
    if bundle_audit.get("verified") is not True or environment.get("verified") is not True:
        raise ValueError("unverified paired inputs")
    if canonical_hash(environment["content"]) != environment["environment_sha256"]:
        raise ValueError("environment hash mismatch")
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("invalid paired cases")
    for key, expected in {"top_k": 40, "request_limit": 200, "timeout_s": 30, "wall_s": 900}.items():
        if type(config[key]) is not int or config[key] != expected:
            raise ValueError("unsupported first-batch budget: " + key)
    if any(c.snapshot_id != bundle_audit["target_snapshot_id"] or c.user_id != config["user_id"] or c.split != "dev" for c in cases):
        raise ValueError("case snapshot/scope/split mismatch")
    common = {"protocol": PROTOCOL, "bundle_sha256": canonical_hash(bundle_audit),
              "snapshot_id": bundle_audit["target_snapshot_id"], "environment_sha256": environment["environment_sha256"],
              "implementation": implementation, "config": config,
              "cases": [{"case_id": c.case_id, "question": c.question, "sha256": canonical_hash(c.model_dump(mode="json"))} for c in cases]}
    arms = {}
    for name in arm_order:
        value = {"common": deepcopy(common), "profile": name, "lexical": deepcopy(_profile(name))}
        arms[name] = {**value, "sha256": canonical_hash(value)}
    result = {"common": deepcopy(common), "arms": arms}
    return {**result, "sha256": canonical_hash(result)}


def assert_paired_comparable(left, right):
    for value in (left, right):
        if value["sha256"] != canonical_hash({k: v for k, v in value.items() if k != "sha256"}):
            raise ValueError("arm fingerprint mismatch")
        if _profile(value.get("profile")) is None or value.get("lexical") != _profile(value.get("profile")):
            raise ValueError("undeclared lexical variation")
    if left["common"] != right["common"]:
        raise ValueError("paired common identity changed")


def validate_identity(identity):
    if identity.get("sha256") != canonical_hash({k: v for k, v in identity.items() if k != "sha256"}):
        raise ValueError("paired identity hash mismatch")
    arm_order = comparison_arms(identity["common"]["config"])
    if set(identity["arms"]) != set(arm_order):
        raise ValueError("paired arm set changed")
    assert_paired_comparable(*(identity["arms"][a] for a in arm_order))
    if any(arm["common"] != identity["common"] or arm["profile"] != name for name, arm in identity["arms"].items()):
        raise ValueError("paired identity common/profile mismatch")


def expected_request(identity, case_id, arm):
    common, config = identity["common"], identity["common"]["config"]
    query = next(c["question"] for c in common["cases"] if c["case_id"] == case_id)
    text = {"match": {"contextualized_content": query}}
    if arm == "ik":
        text = {"multi_match": {"query": query, "fields": ["contextualized_content.zh^2", "contextualized_content"],
                                "type": "best_fields", "tie_breaker": 0.0}}
    elif arm == "ik_max_word":
        # multi_match.analyzer would also change the standard fallback. Expand
        # best_fields into its equivalent dis_max and override only the zh match.
        text = {"dis_max": {"queries": [
            {"match": {"contextualized_content.zh": {"query": query, "analyzer": "ik_max_word", "boost": 2}}},
            {"match": {"contextualized_content": query}}], "tie_breaker": 0.0}}
    elif arm != "standard":
        raise ValueError("unknown lexical arm")
    return {"index": config["index"], "query": {"bool": {"filter": [{"term": {"user_id": config["user_id"]}},
        {"term": {"is_active": True}}, {"term": {"index_generation": config["index_generation"]}}], "must": text}},
        "size": config["top_k"], "source_includes": SOURCE_FIELDS}
