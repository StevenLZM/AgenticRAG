import pytest

from evals.gold_mapping import map_source_facts


def parent(**changes):
    return {"id": "p1", "user_id": "eval-user", "document_version_id": "v1", "status": "active",
            "content": "工程部每周最多 3 天远程办公。", "ast_locator": '{"spans":[{"canonical_path":"#/text_blocks/1"}]}',
            **changes}


def test_maps_frozen_source_anchor_to_scoped_parent_with_ast():
    facts = [{"fact_id": "R02", "anchor_terms": ["工程部每周最多3天远程办公。"]}]
    result = map_source_facts(facts, [parent()], user_id="eval-user", document_version_id="v1")
    assert result["R02"]["parent_id"] == "p1"
    assert result["R02"]["ast_locator"]["spans"][0]["canonical_path"] == "#/text_blocks/1"


@pytest.mark.parametrize("rows", [[], [parent(user_id="other")], [parent(document_version_id="old")],
                                   [parent(status="inactive")], [parent(), parent(id="p2")]])
def test_missing_foreign_stale_or_ambiguous_source_fails(rows):
    with pytest.raises(ValueError, match="R02"):
        map_source_facts([{"fact_id": "R02", "anchor_terms": ["工程部"]}], rows,
                         user_id="eval-user", document_version_id="v1")
