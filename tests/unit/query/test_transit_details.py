"""Transit legs must survive the domain projection, public boundary and replay."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from agentic_rag.query.public_answer import project_public_answer
from agentic_rag.query.tool_answers import build_tool_answer

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/amap_transit_route.json"


def transit_result(payload, *, arguments=None, text=False):
    return {
        "call_id": "transit-1",
        "tool_id": "mcp.amap_maps.maps_direction_transit_integrated",
        "source_kind": "external",
        "status": "success",
        "observed_at": "2026-10-02T08:00:32Z",
        "data": {
            "structured": None if text else payload,
            "text": json.dumps(payload, ensure_ascii=False) if text else "",
            "arguments": arguments or {},
        },
    }


@pytest.mark.parametrize("text", [False, True])
def test_real_transit_segments_preserve_ride_order_stops_and_transfer(text):
    fixture = json.loads(FIXTURE.read_text())
    answer = build_tool_answer(
        [transit_result(fixture["payload"], arguments=fixture["arguments"], text=text)],
        route="fast_rag",
    )
    assert answer is not None
    restored = project_public_answer(
        json.loads(json.dumps(answer)), require_audited=True
    )
    assert restored is not None
    card = restored.model_dump(mode="json", exclude_none=True)["cards"][0]
    steps = card.get("details", {}).get("steps", [])
    assert len(steps) == 5
    assert "1362 米" in steps[0]
    assert all(
        part in steps[1] for part in ("地铁13号线", "龙泽", "上车", "霍营", "下车")
    )
    assert "换乘" in steps[2] and "245 米" in steps[2]
    assert all(part in steps[3] for part in ("换乘", "地铁8号线", "霍营", "金鱼胡同"))
    assert "236 米" in steps[4] and "换乘" not in steps[4]
    assert card["duration_s"] == 4661
    assert "distance_m" not in card  # walking distance is not the full route distance.
    assert "mode=bus" in card["url"]


def test_busline_alternatives_are_not_misrepresented_as_successive_transfers():
    route = {
        "duration": "500",
        "segments": [
            {
                "bus": {
                    "buslines": [
                        {
                            "name": "617路",
                            "departure_stop": {"name": "回龙观桥南"},
                            "arrival_stop": {"name": "育新"},
                        },
                        {
                            "name": "681路",
                            "departure_stop": {"name": "回龙观桥北"},
                            "arrival_stop": {"name": "霍营"},
                        },
                    ]
                }
            }
        ],
    }
    answer = build_tool_answer([transit_result({"transits": [route]})], route="chat")
    steps = answer["cards"][0].get("details", {}).get("steps", [])
    assert len(steps) == 1
    assert all(
        part in steps[0]
        for part in ("617路", "681路", "或", "回龙观桥南", "育新", "回龙观桥北", "霍营")
    )
    assert "换乘" not in steps[0]


def test_malformed_transit_fields_cannot_remove_other_valid_legs_or_invent_stops():
    payload = {
        "transits": [
            {
                "duration": "90",
                "segments": [
                    None,
                    {"walking": "bad", "bus": []},
                    {
                        "walking": {"distance": "NaN", "duration": -5},
                        "bus": {
                            "buslines": [
                                None,
                                {
                                    "name": "8号线",
                                    "departure_stop": [],
                                    "arrival_stop": {"name": "金鱼胡同"},
                                    "private": "never publish",
                                },
                            ]
                        },
                    },
                    {"walking": {"distance": "120"}, "bus": {"buslines": []}},
                ],
            }
        ]
    }
    answer = build_tool_answer([transit_result(payload)], route="chat")
    assert answer is not None
    steps = answer["cards"][0].get("details", {}).get("steps", [])
    assert any("8号线" in step and "金鱼胡同" in step for step in steps)
    assert "120 米" in steps[-1]
    assert "上车" not in str(steps)
    assert all(value not in str(steps) for value in ("NaN", "-5", "never publish"))


def test_long_transit_route_is_bounded_and_explicitly_marks_remaining_legs():
    fixture = json.loads(FIXTURE.read_text())
    payload = deepcopy(fixture["payload"])
    payload["transits"][0]["segments"] *= 6
    answer = build_tool_answer([transit_result(payload)], route="chat")
    assert answer is not None
    steps = answer["cards"][0].get("details", {}).get("steps", [])
    assert 1 < len(steps) <= 12
    assert "高德地图" in steps[-1] and "后续" in steps[-1]


def test_long_transit_labels_do_not_discard_the_entire_route():
    payload = {
        "transits": [
            {
                "duration": "500",
                "segments": [
                    {
                        "bus": {
                            "buslines": [
                                {
                                    "name": "线" * 500,
                                    "departure_stop": {"name": "起" * 500},
                                    "arrival_stop": {"name": "终" * 500},
                                }
                            ]
                        }
                    }
                ],
            }
        ]
    }
    answer = build_tool_answer([transit_result(payload)], route="chat")
    assert answer is not None
    steps = answer["cards"][0].get("details", {}).get("steps", [])
    assert steps and all(len(step) <= 512 for step in steps)
    assert "…" in steps[0]
