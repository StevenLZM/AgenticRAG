"""Bind pre-labelled facts to ingestion output, never to query retrieval results."""
import json
import re
import unicodedata


def _normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def map_source_facts(facts, parents, *, user_id, document_version_id):
    scoped = [p for p in parents if p["user_id"] == user_id
              and p["document_version_id"] == document_version_id and p["status"] == "active"]
    mapped = {}
    for fact in facts:
        anchors = fact["anchor_terms"]
        if not anchors or any(not _normalize(anchor) for anchor in anchors):
            raise ValueError(f"invalid anchors for {fact['fact_id']}")
        matches = [p for p in scoped if all(_normalize(a) in _normalize(p["content"]) for a in anchors)]
        if len(matches) != 1:
            raise ValueError(f"source fact {fact['fact_id']} matched {len(matches)} parents; manual source review required")
        parent = matches[0]
        locator = json.loads(parent["ast_locator"]) if isinstance(parent["ast_locator"], str) else parent["ast_locator"]
        if not isinstance(locator, dict) or not locator:
            raise ValueError(f"source fact {fact['fact_id']} lacks AST provenance")
        if fact["fact_id"] in mapped:
            raise ValueError(f"duplicate source fact {fact['fact_id']}")
        mapped[fact["fact_id"]] = {"parent_id": parent["id"], "user_id": user_id,
                                   "document_version_id": document_version_id, "ast_locator": locator}
    return mapped
