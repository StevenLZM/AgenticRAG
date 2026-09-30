import pytest


def test_anchor_binding_accepts_equivalent_passages_without_requiring_all():
    from evals.formal_gold import bind_anchor
    parents = [{"id": "p1", "document_version_id": "v", "content": "住宿上限 470 元"},
               {"id": "p2", "document_version_id": "v", "content": "住宿上限470元"}]
    children = [{"child_id": "c1", "parent_id": "p1", "content": "住宿上限470元"}]
    group = bind_anchor("v", "住宿上限470元", parents, children)
    assert group["parent_ids_any_of"] == ["p1", "p2"]
    assert group["child_ids_any_of"] == ["c1"]


def test_unmapped_anchor_is_explicit_gap_not_synthetic_qrel():
    from evals.formal_gold import bind_anchor
    result = bind_anchor("v", "上限470元", [], [])
    assert result["mapping_status"] == "unmapped"
    assert result["parent_ids_any_of"] == []


def test_source_claim_must_exist_in_original():
    from evals.formal_gold import check_anchor
    check_anchor("本年度容量为5200份。", "容量为5200份")
    with pytest.raises(ValueError, match="original"):
        check_anchor("本年度容量为5200份。", "容量为6200份")


def test_company_family_split_does_not_leak_years():
    from evals.formal_gold import split_for
    assert split_for("S001-2025-policy") == split_for("S001-2026-product")
    assert split_for("S050-2025-policy") == "test"


def test_heading_is_part_of_parent_evidence_but_not_invented_child_evidence():
    from evals.formal_gold import bind_anchor
    result = bind_anchor("v", ["本科", "哈尔滨工业大学"],
        [{"id": "p", "document_version_id": "v", "heading_path": ["本科 哈尔滨工业大学"],
          "content": "计算机科学与技术专业"}], [])
    assert result["parent_ids_any_of"] == ["p"]
    assert result["mapping_status"] == "parent_only"


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), 90])
def test_non_finite_or_wrong_cached_value_fails_closed(bad):
    from evals.formal_gold import check_cached_number
    with pytest.raises(ValueError, match="cached"):
        check_cached_number(bad, 91)


def test_production_locator_maps_markdown_table_without_changing_child_text():
    # Catches literal-only matching of serialized table rows, not a chunking bug.
    from evals.formal_gold import bind_anchor
    parent = {"id": "p", "document_version_id": "v", "content": "所有数据为合成观测，不\n代表真实经营。"}
    child = {"child_id": "c", "parent_id": "p", "document_version_id": "v",
             "content": "| 所有数据为合成观测，不 |\n| 代表真实经营。 |",
             "ast_locator": {"spans": [{"parent_char_from": 0, "parent_char_to": len(parent["content"])}]}}
    result = bind_anchor("v", "所有数据为合成观测，不代表真实经营。", [parent], [child])
    assert result["mapping_status"] == "mapped"
    assert result["child_ids_any_of"] == ["c"]
    assert result["child_evidence_sets_any_of"] == [["c"]]


def test_cross_child_requires_both_chunks_and_does_not_fill_a_gap():
    from evals.formal_gold import bind_anchor
    parent = {"id": "p", "document_version_id": "v", "content": "上限470元，须提前审批。"}
    def child(key, start, end):
        return {"child_id": key, "parent_id": "p", "document_version_id": "v",
                "content": parent["content"][start:end],
                "ast_locator": {"spans": [{"parent_char_from": start, "parent_char_to": end}]}}
    result = bind_anchor("v", parent["content"], [parent], [child("a", 0, 6), child("b", 6, 13)])
    assert result["child_ids_any_of"] == []
    assert result["child_evidence_sets_any_of"] == [["a", "b"]]
    assert result["mapping_status"] == "mapped_multi_child"
    gap = bind_anchor("v", parent["content"], [parent], [child("a", 0, 6), child("b", 7, 13)])
    assert gap["mapping_status"] == "parent_only"


def test_heading_fact_uses_only_actual_child_heading_context():
    from evals.formal_gold import bind_anchor
    parent = {"id": "p", "document_version_id": "v", "content": "计算机专业", "heading_path": ["本科 哈尔滨工业大学"]}
    child = {"child_id": "c", "parent_id": "p", "document_version_id": "v", "content": "计算机专业",
             "heading_path": ["本科 哈尔滨工业大学"], "contextualized_content": "本科 哈尔滨工业大学\n计算机专业"}
    assert bind_anchor("v", ["本科", "哈尔滨工业大学"], [parent], [child])["child_ids_any_of"] == ["c"]
    child.pop("heading_path")
    child.pop("contextualized_content")
    assert bind_anchor("v", ["本科", "哈尔滨工业大学"], [parent], [child])["mapping_status"] == "parent_only"


def test_malformed_or_foreign_locator_never_creates_child_qrel():
    from evals.formal_gold import bind_anchor
    p = {"id": "p", "document_version_id": "v", "content": "上限470元"}
    cs = [{"child_id": "bad", "parent_id": "p", "document_version_id": "other", "content": "上限470元"},
          {"child_id": "wide", "parent_id": "p", "document_version_id": "v", "content": " unrelated ",
           "ast_locator": {"spans": [{"parent_char_from": -1, "parent_char_to": 999}]}}]
    result = bind_anchor("v", "上限470元", [p], cs)
    assert result["child_ids_any_of"] == []
    assert result["mapping_status"] == "parent_only"


def test_normal_chunk_boundary_may_omit_only_whitespace():
    # Real HybridChunker omits a separating space from both adjacent spans.
    from evals.formal_gold import bind_anchor
    parent = {"id": "p", "document_version_id": "v", "content": "甲乙 丙丁"}
    children = [{"child_id": "a", "parent_id": "p", "content": "甲乙",
                 "ast_locator": {"spans": [{"parent_char_from": 0, "parent_char_to": 2}]}},
                {"child_id": "b", "parent_id": "p", "content": "丙丁",
                 "ast_locator": {"spans": [{"parent_char_from": 3, "parent_char_to": 5}]}}]
    result = bind_anchor("v", "甲乙 丙丁", [parent], children)
    assert result["child_evidence_sets_any_of"] == [["a", "b"]]
    assert result["mapping_status"] == "mapped_multi_child"
    parent["content"] = "甲乙X丙丁"
    gap = bind_anchor("v", "甲乙X丙丁", [parent], children)
    assert gap["mapping_status"] == "parent_only"
