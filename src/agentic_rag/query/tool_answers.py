"""Deterministic domain projection of tool observations, never model-written facts.

AMap fields and units follow its place-search and route Web Service schemas.
URI links use the documented marker/navigation API and GCJ-02 coordinates.
No upstream URL, raw result, or arbitrary provider prose crosses this boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
import json
import math
from typing import Any
from urllib.parse import urlencode

from pydantic import BaseModel, ValidationError

from agentic_rag.query.public_answer import (
    PublicExternalSource,
    PublicPlaceCard,
    PublicRouteCard,
    normalize_location,
    project_public_answer,
    valid_poi_id,
)

_PLACE_TOOLS = {"maps_text_search", "maps_around_search", "maps_search_detail"}
_ROUTE_MODES = {
    "maps_direction_driving": ("driving", "car", "驾车"),
    "maps_direction_walking": ("walking", "walk", "步行"),
    "maps_direction_bicycling": ("bicycling", "ride", "骑行"),
    "maps_bicycling": ("bicycling", "ride", "骑行"),
    "maps_direction_transit_integrated": ("transit", "bus", "公交"),
}
_MAX_PAYLOAD_BYTES = 131_072


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if (
        not value
        or len(value) > 512
        or any(ord(c) < 32 and c not in "\n\t" for c in value)
    ):
        return None
    return value


def _metric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _payload(data: Mapping[str, Any]) -> dict[str, Any] | list[Any] | None:
    try:
        if (
            len(json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            > _MAX_PAYLOAD_BYTES
        ):
            return None
        payload = data.get("structured")
        if payload is None:
            text = data.get("text")
            if not isinstance(text, str):
                return None
            payload = json.loads(text)
        for _ in range(3):
            if not isinstance(payload, dict):
                break
            if (
                payload.get("status") in ("0", 0, False)
                or payload.get("success") is False
            ):
                return None
            # Known wrappers used by MCP structuredContent, never recursively scan prose.
            nested = payload.get("data", payload.get("result"))
            if isinstance(nested, (dict, list)) and not any(
                k in payload for k in ("pois", "route", "paths", "transits", "name")
            ):
                payload = nested
            else:
                break
        return payload if isinstance(payload, (dict, list)) else None
    except (ValueError, TypeError, RecursionError, UnicodeError):
        return None


def _endpoint(value: object) -> dict[str, str] | None:
    endpoint: dict[str, str] = {}
    if isinstance(value, Mapping):
        name = _text(value.get("name"))
        if name:
            endpoint["name"] = name
        location = normalize_location(value.get("location"))
    else:
        location = normalize_location(value)
    if location:
        endpoint.update(location=location, coordinate_system="GCJ-02")
    return endpoint or None


def _place_cards(payload: Any, source_id: str) -> tuple[list[dict[str, Any]], bool]:
    pois = payload if isinstance(payload, list) else payload.get("pois")
    if pois is None and isinstance(payload, dict) and "name" in payload:
        pois = [payload]
    if not isinstance(pois, list):
        return [], False
    cards = []
    for index, poi in enumerate(pois[:32]):
        if not isinstance(poi, dict) or not (name := _text(poi.get("name"))):
            continue
        card: dict[str, Any] = {
            "id": f"{source_id}-p{index}",
            "kind": "place",
            "title": name,
            "source_id": source_id,
        }
        if valid_poi_id(poi.get("id")):
            card["poi_id"] = poi["id"]
        if address := _text(poi.get("address")):
            card["address"] = address
        if location := normalize_location(poi.get("location")):
            card.update(location=location, coordinate_system="GCJ-02")
            card["url"] = "https://uri.amap.com/marker?" + urlencode(
                {
                    "position": location,
                    "name": name,
                    "src": "agentic_rag",
                    "coordinate": "gaode",
                    "callnative": "0",
                }
            )
        if (distance := _metric(poi.get("distance"))) is not None:
            card["distance_m"] = distance
        if "url" not in card and "poi_id" in card:
            card["url"] = "https://uri.amap.com/marker?" + urlencode(
                {
                    "poiid": card["poi_id"],
                    "src": "agentic_rag",
                    "callnative": "0",
                }
            )
        details = {}
        for field, raw in (
            ("category", "type"),
            ("city", "cityname"),
            ("district", "adname"),
        ):
            if text := _text(poi.get(raw)):
                details[field] = text
        business = poi.get("business", poi.get("biz_ext"))
        if isinstance(business, dict) and (
            hours := _text(business.get("opentime_week"))
        ):
            details["opening_hours"] = hours
        if details:
            card["details"] = details
        cards.append(
            PublicPlaceCard.model_validate(card).model_dump(
                mode="json", exclude_none=True
            )
        )
    return cards, not pois


def _transit_label(value: object) -> str | None:
    """Leave room for a line and both stops inside the public text bound."""
    text = _text(value)
    return text if text is None or len(text) <= 120 else text[:119] + "…"


def _transit_ride(bus: object) -> str | None:
    lines = bus.get("buslines") if isinstance(bus, dict) else None
    if not isinstance(lines, list):
        return None
    alternatives: list[str] = []
    for line in lines:
        if not isinstance(line, dict) or not (name := _transit_label(line.get("name"))):
            continue
        departure = line.get("departure_stop")
        arrival = line.get("arrival_stop")
        stops = []
        for value, action in ((departure, "上车"), (arrival, "下车")):
            stop = (
                _transit_label(value.get("name")) if isinstance(value, dict) else None
            )
            if stop:
                stops.append(f"{stop} {action}")
        option = name + ("：" + " → ".join(stops) if stops else "")
        # Entries in buslines are alternatives for this leg, not sequential
        # transfers. Bound the combined choice without losing the whole route.
        if len(" 或 ".join([*alternatives, option])) > 460:
            return " 或 ".join(alternatives) + "（更多备选线路请在高德地图中查看）"
        alternatives.append(option)
    return " 或 ".join(alternatives) or None


def _transit_steps(segments: object) -> list[str]:
    if not isinstance(segments, list):
        return []
    steps: list[str] = []
    has_ridden = False
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        ride = _transit_ride(segment.get("bus"))
        walking = segment.get("walking")
        if isinstance(walking, dict):
            distance = _metric(walking.get("distance"))
            duration = _metric(walking.get("duration"))
            prefix = "换乘步行" if has_ridden and ride else "步行"
            if distance is not None:
                steps.append(f"{prefix} {distance:g} 米")
            elif duration is not None:
                steps.append(f"{prefix}，预计 {duration:g} 秒")
        if ride:
            steps.append(("换乘 " if has_ridden else "乘坐 ") + ride)
            has_ridden = True
        if len(steps) > 12:
            return [*steps[:11], "后续路段请在高德地图中查看。"]
    return steps


def _route_cards(
    payload: Any, arguments: Mapping[str, Any], source_id: str, tool: str
) -> tuple[list[dict[str, Any]], bool]:
    route = payload.get("route", payload) if isinstance(payload, dict) else {}
    if not isinstance(route, dict):
        return [], False
    paths = (
        payload
        if isinstance(payload, list)
        else route.get("paths", route.get("transits"))
    )
    if not isinstance(paths, list):
        return [], False
    mode, uri_mode, label = _ROUTE_MODES[tool]
    origin = _endpoint(route.get("origin", arguments.get("origin")))
    destination = _endpoint(route.get("destination", arguments.get("destination")))
    cards = []
    for index, path in enumerate(paths[:8]):
        if not isinstance(path, dict):
            continue
        cost = path.get("cost")
        distance = _metric(path.get("distance"))
        duration = _metric(
            path.get(
                "duration", cost.get("duration") if isinstance(cost, dict) else None
            )
        )
        if distance is None and duration is None:
            continue
        card: dict[str, Any] = {
            "id": f"{source_id}-r{index}",
            "kind": "route",
            "title": f"{label}路线 {index + 1}",
            "source_id": source_id,
            "mode": mode,
        }
        for key, value in (
            ("origin", origin),
            ("destination", destination),
            ("distance_m", distance),
            ("duration_s", duration),
        ):
            if value is not None:
                card[key] = value
        if (
            origin
            and origin.get("location")
            and destination
            and destination.get("location")
        ):
            card["url"] = "https://uri.amap.com/navigation?" + urlencode(
                {
                    "from": origin["location"],
                    "to": destination["location"],
                    "mode": uri_mode,
                    "src": "agentic_rag",
                    "callnative": "0",
                }
            )
        safe_steps = _transit_steps(path.get("segments")) if mode == "transit" else []
        steps = path.get("steps")
        if not safe_steps and isinstance(steps, list):
            safe_steps = [
                text
                for step in steps[:12]
                if isinstance(step, dict) and (text := _text(step.get("instruction")))
            ]
        if safe_steps:
            card["details"] = {"steps": safe_steps}
        cards.append(
            PublicRouteCard.model_validate(card).model_dump(
                mode="json", exclude_none=True
            )
        )
    return cards, not paths


def _geocode_cards(
    payload: Any, arguments: Mapping[str, Any], source_id: str, *, reverse: bool
) -> tuple[list[dict[str, Any]], bool]:
    if reverse:
        geocode = (
            payload.get("regeocode", payload) if isinstance(payload, dict) else None
        )
        geocodes = [geocode] if isinstance(geocode, dict) else None
    else:
        geocodes = (
            payload
            if isinstance(payload, list)
            else payload.get("geocodes", payload.get("results"))
        )
    if not isinstance(geocodes, list):
        return [], False
    pois = []
    for geocode in geocodes[:32]:
        if not isinstance(geocode, dict):
            continue
        address = _text(geocode.get("formatted_address"))
        # DashScope's maps_geo results omit formatted_address. The validated
        # query can label a returned coordinate, but is never a verified address.
        name = address
        if not name and not reverse and normalize_location(geocode.get("location")):
            name = _text(arguments.get("address"))
        if not name:
            continue
        components = geocode.get("addressComponent", {}) if reverse else geocode
        if not isinstance(components, dict):
            components = {}
        pois.append(
            {
                "name": name,
                "address": address,
                "location": geocode.get(
                    "location", arguments.get("location") if reverse else None
                ),
                "cityname": components.get("city"),
                "adname": components.get("district"),
            }
        )
    cards, _ = _place_cards(pois, source_id)
    return cards, not geocodes


def _distance_cards(
    payload: Any, arguments: Mapping[str, Any], source_id: str
) -> tuple[list[dict[str, Any]], bool]:
    results = payload if isinstance(payload, list) else payload.get("results")
    if not isinstance(results, list):
        return [], False
    origins = arguments.get("origins")
    points = origins.split("|") if isinstance(origins, str) else []
    mode = {"1": "maps_direction_driving", "3": "maps_direction_walking"}.get(
        str(arguments.get("type"))
    )
    cards: list[dict[str, Any]] = []
    for index, result in enumerate(results[:32]):
        if not isinstance(result, dict) or result.get("code") not in (None, "", "0", 0):
            continue
        origin_id = str(result.get("origin_id", ""))
        origin = (
            points[int(origin_id) - 1]
            if origin_id.isdigit() and 1 <= int(origin_id) <= len(points)
            else None
        )
        destination = (
            arguments.get("destination")
            if str(result.get("dest_id", "")) == "1"
            else None
        )
        if mode:
            projected, _ = _route_cards(
                {"origin": origin, "destination": destination, "paths": [result]},
                {},
                source_id,
                mode,
            )
            if projected:
                projected[0]["id"] = f"{source_id}-d{index}"
                cards.extend(projected)
            continue
        distance = _metric(result.get("distance"))
        if distance is None:
            continue
        title = "直线距离" if str(arguments.get("type")) == "0" else "距离结果"
        card: dict[str, Any] = {
            "id": f"{source_id}-d{index}",
            "kind": "route",
            "title": f"{title} {index + 1}",
            "source_id": source_id,
            "distance_m": distance,
        }
        if (duration := _metric(result.get("duration"))) is not None:
            card["duration_s"] = duration
        for key, point in (("origin", origin), ("destination", destination)):
            if endpoint := _endpoint(point):
                card[key] = endpoint
        cards.append(
            PublicRouteCard.model_validate(card).model_dump(
                mode="json", exclude_none=True
            )
        )
    return cards, not results


def _card_text(card: Mapping[str, Any]) -> str:
    pieces = [card["title"]]
    if card.get("address"):
        pieces.append(f"地址：{card['address']}")
    if "distance_m" in card:
        label = "距搜索中心" if card["kind"] == "place" else "距离"
        pieces.append(f"{label} {card['distance_m']:g} 米")
    if "duration_s" in card:
        pieces.append(f"预计耗时 {card['duration_s']:g} 秒")
    return "；".join(pieces) + "。"


def build_tool_answer(
    results: Iterable[object],
    *,
    route: str,
    document_answer: Mapping[str, Any] | None = None,
    max_cards: int = 8,
) -> dict[str, Any] | None:
    """Project successful allowlisted facts and optionally merge audited documents.

    Accepts runtime ToolResult models and their checkpoint JSON representation.
    Empty searches are observations; unsupported/malformed successes are not.
    """
    if type(max_cards) is not int or not 1 <= max_cards <= 32:
        raise ValueError("max_cards must be an integer from 1 to 32")
    base: dict[str, Any] = {}
    if document_answer is not None:
        document = project_public_answer(
            document_answer, route=route, require_audited=True
        )
        if (
            document is None
            or document.audited is not True
            or document.status is not None
        ):
            return None
        base = document.model_dump(mode="json", exclude_none=True)
    sources: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    seen: set[str] = set()
    incomplete = False
    for index, result in enumerate(results):
        if index >= 64 or len(sources) >= 16:
            incomplete = True
            break
        item = (
            result.model_dump(mode="json") if isinstance(result, BaseModel) else result
        )
        if not isinstance(item, Mapping) or item.get("status") != "success":
            incomplete = True
            continue
        tool_id = item.get("tool_id")
        call_id = item.get("call_id")
        data = item.get("data")
        if (
            not isinstance(tool_id, str)
            or not isinstance(call_id, str)
            or not isinstance(data, Mapping)
        ):
            incomplete = True
            continue
        if item.get("source_kind") == "document":
            continue
        source_id = "tool-" + sha256(f"{call_id}\0{tool_id}".encode()).hexdigest()[:24]
        if source_id in seen:
            continue
        is_calculation = (
            tool_id == "local.calculator" and item.get("source_kind") == "calculation"
        )
        is_amap = (
            tool_id.startswith("mcp.amap_maps.")
            and item.get("source_kind") == "external"
        )
        if not is_calculation and not is_amap:
            incomplete = True
            continue
        tool = tool_id.removeprefix("mcp.amap_maps.")
        if is_amap and tool not in _PLACE_TOOLS | _ROUTE_MODES.keys() | {
            "maps_geo",
            "maps_regeo",
            "maps_distance",
        }:
            # Successful intermediate tools may aid planning without a public domain projection.
            continue
        try:
            source = PublicExternalSource.model_validate(
                {
                    "id": source_id,
                    "tool_id": tool_id,
                    "provider": "本地计算" if is_calculation else "高德地图",
                    "observed_at": item.get("observed_at"),
                }
            ).model_dump(mode="json")
            new_cards: list[dict[str, Any]] = []
            if is_calculation:
                value = data.get("value")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    incomplete = True
                    continue
                text = f"计算结果：{value}。"
            else:
                payload = _payload(data)
                if payload is None:
                    incomplete = True
                    continue
                arguments = data.get("arguments")
                arguments = arguments if isinstance(arguments, Mapping) else {}
                if tool in _PLACE_TOOLS:
                    new_cards, empty = _place_cards(payload, source_id)
                    text = "高德地图未找到匹配的地点。" if empty else ""
                elif tool in _ROUTE_MODES:
                    new_cards, empty = _route_cards(payload, arguments, source_id, tool)
                    text = "高德地图未找到匹配的路线。" if empty else ""
                elif tool in {"maps_geo", "maps_regeo"}:
                    new_cards, empty = _geocode_cards(
                        payload, arguments, source_id, reverse=tool == "maps_regeo"
                    )
                    text = "高德地图未找到匹配的地点。" if empty else ""
                elif tool == "maps_distance":
                    new_cards, empty = _distance_cards(payload, arguments, source_id)
                    text = "高德地图未返回距离结果。" if empty else ""
                else:
                    incomplete = True
                    continue
                available = max_cards - len(cards)
                if new_cards and available <= 0:
                    continue
                new_cards = new_cards[:available]
                if new_cards:
                    text = "\n".join(_card_text(card) for card in new_cards)
                if not text:
                    incomplete = True
                    continue
            sources.append(source)
            cards.extend(new_cards)
            # A segment remains bounded even for a page of long place addresses.
            for offset in range(0, len(text), 8000):
                segments.append(
                    {
                        "kind": "content",
                        "text": text[offset : offset + 8000],
                        "evidence_ids": [],
                    }
                )
            seen.add(source_id)
        except (ValidationError, ValueError, TypeError, OverflowError):
            incomplete = True
    if not segments:
        return None
    if incomplete:
        segments.append(
            {
                "kind": "content",
                "text": "部分工具未返回可核验结果，以上信息可能不完整。",
                "evidence_ids": [],
            }
        )
    combined = {
        **base,
        "route": route,
        "tool_audited": True,
        "segments": [*base.get("segments", []), *segments],
        "external_sources": sources,
        "cards": cards,
    }
    answer = project_public_answer(combined, require_audited=True)
    return answer.model_dump(mode="json", exclude_none=True) if answer else None
