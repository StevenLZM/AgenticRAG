"""Original-grounded gold helpers, independent from retrieval results and models."""
import re
import math
import json
import unicodedata


def normalize(text):
    return "".join(unicodedata.normalize("NFKC", str(text)).split())


def check_anchor(original, anchor):
    if not normalize(anchor) or normalize(anchor) not in normalize(original):
        raise ValueError("gold anchor absent from original")


def split_for(key):
    match = re.match(r"S(\d+)-", key)
    if not match:
        return "test"
    company = int(match[1])
    return "dev" if company <= 30 else "validation" if company <= 40 else "test"


def bind_anchor(version, anchor, parents, children):
    anchors = [anchor] if isinstance(anchor, str) else anchor
    def matches(text):
        return all(normalize(a) in normalize(text) for a in anchors)
    ps = [p for p in parents if p["document_version_id"] == version and matches(
        " ".join(p.get("heading_path", [])) + " " + p["content"])]
    singles, evidence_sets, methods = set(), [], set()
    for parent in ps:
        scoped = [c for c in children if c["parent_id"] == parent["id"]
                  and c.get("document_version_id", version) == version]
        body_ranges, heading_anchors = _anchor_ranges(parent, anchors)
        proven = []
        for child in scoped:
            text = child.get("contextualized_content") or child["content"]
            ranges = _child_ranges(child, len(parent["content"]))
            if matches(text):
                singles.add(child["child_id"])
                methods.add("rendered_text")
            elif (body_ranges or heading_anchors) and _heading_matches(child, heading_anchors) and all(
                _range_covered(target, ranges) for target in body_ranges
            ):
                singles.add(child["child_id"])
                methods.add("production_ast_locator")
            proven.append((child, ranges))
        # A set is sufficient only when all source characters are covered. Do not
        # infer continuity from a min/max envelope or join unrelated text chunks.
        if not any(c["child_id"] in singles for c in scoped):
            combination = _cover_with_children(body_ranges, heading_anchors, proven)
            if combination:
                evidence_sets.append(combination)
                methods.add("production_ast_locator_union")
    evidence_sets = [[key] for key in sorted(singles)] + evidence_sets
    return {"source_anchors": anchors, "parent_ids_any_of": sorted(p["id"] for p in ps),
            "child_ids_any_of": sorted(singles), "child_evidence_sets_any_of": evidence_sets,
            "child_mapping_methods": sorted(methods),
            "mapping_status": "mapped" if singles else "mapped_multi_child" if evidence_sets else "parent_only" if ps else "unmapped"}


def _anchor_ranges(parent, anchors):
    text, positions = [], []
    for i, char in enumerate(parent["content"]):
        for normalized in unicodedata.normalize("NFKC", char):
            if not normalized.isspace():
                text.append(normalized)
                positions.append(i)
    flat = "".join(text)
    ranges, headings = [], []
    for anchor in anchors:
        needle = normalize(anchor)
        start = flat.find(needle)
        if start >= 0 and needle:
            # HybridChunker can omit a separator space at adjacent boundaries.
            # Require every substantive original character, not those normalized
            # away by the same anchor rule. Missing punctuation/letters still fail.
            run_start = run_end = None
            for pos in sorted(set(positions[start:start + len(needle)])):
                if run_end is not None and pos != run_end:
                    ranges.append((run_start, run_end))
                    run_start = None
                if run_start is None:
                    run_start = pos
                run_end = pos + 1
            if run_start is not None:
                ranges.append((run_start, run_end))
        elif needle and needle in normalize(" ".join(parent.get("heading_path", []))):
            headings.append(anchor)
        else:
            return [], []
    return ranges, headings


def _child_ranges(child, parent_length):
    locator = child.get("ast_locator")
    if isinstance(locator, str):
        try:
            locator = json.loads(locator)
        except ValueError:
            return []
    if not isinstance(locator, dict) or not isinstance(locator.get("spans"), list):
        return []
    ranges = []
    for span in locator["spans"]:
        if not isinstance(span, dict):
            return []
        start, end = span.get("parent_char_from"), span.get("parent_char_to")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= parent_length:
            return []
        ranges.append((start, end))
    return sorted(ranges)


def _heading_matches(child, anchors):
    heading = normalize(" ".join(child.get("heading_path", [])))
    context = normalize(child.get("contextualized_content", ""))
    return all(normalize(a) in heading and normalize(a) in context for a in anchors)


def _range_covered(target, ranges):
    cursor, end = target
    for start, stop in sorted(ranges):
        if start > cursor:
            break
        cursor = max(cursor, stop)
        if cursor >= end:
            return True
    return False


def _cover_with_children(targets, headings, children):
    selected = set()
    for start, end in targets:
        cursor = start
        while cursor < end:
            choices = [(stop, child["child_id"]) for child, ranges in children
                       for begin, stop in ranges if begin <= cursor < stop]
            if not choices:
                return []
            cursor, key = max(choices)
            selected.add(key)
    for anchor in headings:
        choices = [child["child_id"] for child, _ in children if _heading_matches(child, [anchor])]
        if not choices:
            return []
        selected.add(sorted(choices)[0])
    return sorted(selected)


def check_cached_number(actual, expected):
    if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError("spreadsheet cached result mismatch")
    if abs(actual - expected) > 1e-12:
        raise ValueError("spreadsheet cached result mismatch")
