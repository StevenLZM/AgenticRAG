"""Independent source audits and typed map projection at the public boundary."""

from copy import deepcopy

import pytest

from agentic_rag.query.public_answer import project_public_answer


def answer(route="chat"):
    return {
        "route": route,
        "tool_audited": True,
        "segments": [{"kind": "content", "text": "西湖"}],
        "external_sources": [
            {
                "id": "s1",
                "tool_id": "mcp.amap_maps.maps_text_search",
                "provider": "高德地图",
                "observed_at": "2026-10-02T00:00:00Z",
            }
        ],
        "cards": [
            {
                "id": "c1",
                "source_id": "s1",
                "kind": "place",
                "title": "西湖",
                "location": "120.15,30.25",
                "coordinate_system": "GCJ-02",
                "url": "https://uri.amap.com/marker?position=120.15%2C30.25&coordinate=gaode",
            }
        ],
    }


@pytest.mark.parametrize("route", ["chat", "fast_rag", "research"])
def test_tool_only_answer_can_pass_source_audit_for_each_existing_route(route):
    projected = project_public_answer(answer(route), require_audited=True)
    assert projected is not None
    assert projected.model_dump(mode="json")["cards"][0]["title"] == "西湖"
    assert projected.audited is None


def test_chat_cannot_bypass_tool_audit():
    value = answer()
    value.pop("tool_audited")
    assert project_public_answer(value) is None


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "http://uri.amap.com/marker?position=120,30",
        "https://uri.amap.com.evil.com/marker?position=120,30",
        "https://user@uri.amap.com/marker?position=120,30",
        "https://uri.amap.com/redirect?url=https://evil.com",
        "https://uri.amap.com/marker?position=120,30&url=https://evil.com",
        "https://uri.amap.com/marker?position=181,95",
    ],
)
def test_unsafe_or_invalid_map_links_fail_closed(url):
    value = answer()
    value["cards"][0]["url"] = url
    assert project_public_answer(value) is None


def test_mixed_document_evidence_still_requires_document_audit():
    value = answer("research")
    value["segments"][0]["evidence_ids"] = ["e1"]
    value["evidence_parent_ids"] = ["p1"]
    assert project_public_answer(value, require_audited=True) is None
    value["audited"] = True
    assert project_public_answer(value, require_audited=True) is not None


def test_chat_with_real_document_evidence_accepts_its_separate_document_audit():
    value = answer("chat")
    value["audited"] = True
    value["segments"][0]["evidence_ids"] = ["e1"]
    value["evidence_parent_ids"] = ["p1"]
    assert project_public_answer(value, require_audited=True) is not None


def test_card_source_must_exist_and_location_must_have_correct_coordinate_system():
    for change in (
        {"source_id": "missing"},
        {"coordinate_system": "WGS-84"},
        {"location": "NaN,30"},
    ):
        value = deepcopy(answer())
        value["cards"][0].update(change)
        assert project_public_answer(value) is None


def test_projection_discards_raw_payload_at_every_new_boundary():
    value = answer()
    value["cards"][0]["raw"] = "secret"
    value["cards"][0]["details"] = {"city": "杭州", "provider_response": "secret"}
    value["external_sources"][0]["headers"] = {"authorization": "secret"}
    projected = project_public_answer(value)
    assert projected is not None
    assert "secret" not in projected.model_dump_json()


def test_poi_id_marker_link_must_match_card_reference():
    value = answer()
    card = value["cards"][0]
    card.pop("location")
    card.pop("coordinate_system")
    card.update(
        poi_id="B023B08WDR",
        url="https://uri.amap.com/marker?poiid=B023B08WDR&src=agentic_rag&callnative=0",
    )
    assert project_public_answer(value) is not None
    card["url"] = "https://uri.amap.com/marker?poiid=B023B08WDX"
    assert project_public_answer(value) is None
