"""Strict public projection for audited answers and safe terminal outcomes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
import re
from typing import Annotated, Any, Literal
from urllib.parse import parse_qs, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
    SerializerFunctionWrapHandler,
)


SafeIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    ),
]
PublicTerminalStatus = Literal[
    "clarify",
    "refuse",
    "cannot_answer",
    "research_action_invalid",
    "audit_failed",
    "research_round_limit",
]
PublicRoute = Literal["chat", "fast_rag", "research"]
PublicClientProvenance = Literal["api", "fixture", "real_query_api", "real_query_graph"]
CardText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
]
Metric = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
PoiIdentifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9]{1,64}$")]


def valid_poi_id(value: object) -> bool:
    return (
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9]{1,64}", value) is not None
    )


def normalize_location(value: object) -> str | None:
    """Validate explicit longitude,latitude in GCJ-02; never infer or swap axes."""
    if not isinstance(value, str) or len(value) > 80:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        return None
    try:
        longitude, latitude = (Decimal(part.strip()) for part in parts)
        if not (longitude.is_finite() and latitude.is_finite()):
            return None
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            return None
        # Avoid expanding adversarial exponents into unbounded strings.
        if any(abs(int(n.as_tuple().exponent)) > 12 for n in (longitude, latitude)):
            return None
        return ",".join(format(n.normalize(), "f") for n in (longitude, latitude))
    except (InvalidOperation, ValueError):
        return None


def is_safe_amap_url(value: str) -> bool:
    """Only official marker/navigation URLs with bounded, known query fields."""
    if len(value) > 4096 or any(ord(c) < 32 for c in value):
        return False
    try:
        url = urlsplit(value)
        if url.scheme != "https" or url.netloc != "uri.amap.com" or url.fragment:
            return False
        params = parse_qs(url.query, strict_parsing=True, keep_blank_values=True)
        if any(len(v) != 1 for v in params.values()):
            return False
        if params.get("coordinate", ["gaode"])[0] != "gaode":
            return False
        if params.get("callnative", ["0"])[0] != "0":
            return False
        if url.path == "/marker":
            if "poiid" in params:
                return not (
                    params.keys() - {"poiid", "src", "callnative"}
                ) and valid_poi_id(params["poiid"][0])
            return (
                not (
                    params.keys()
                    - {"position", "name", "src", "coordinate", "callnative"}
                )
                and normalize_location(params.get("position", [None])[0]) is not None
            )
        if url.path == "/navigation":
            return (
                not (
                    params.keys()
                    - {"from", "to", "mode", "src", "coordinate", "callnative"}
                )
                and all(
                    normalize_location(params.get(k, [None])[0]) for k in ("from", "to")
                )
                and params.get("mode", [None])[0] in {"car", "walk", "ride", "bus"}
            )
    except (ValueError, TypeError):
        return False
    return False


class PublicExternalSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: SafeIdentifier
    tool_id: SafeIdentifier
    provider: Literal["高德地图", "本地计算"]
    observed_at: Annotated[str, StringConstraints(min_length=1, max_length=64)]

    @field_validator("observed_at")
    @classmethod
    def _valid_time(cls, value: str) -> str:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if observed.tzinfo is None:
            raise ValueError("source observation requires an explicit timezone")
        return value

    @model_validator(mode="after")
    def _known_provider(self) -> "PublicExternalSource":
        if self.provider == "高德地图" and not self.tool_id.startswith(
            "mcp.amap_maps."
        ):
            raise ValueError("map provenance must name the configured map service")
        if self.provider == "本地计算" and self.tool_id != "local.calculator":
            raise ValueError("calculation provenance must name the local calculator")
        return self


class PublicMapEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: CardText | None = None
    location: str | None = None
    coordinate_system: Literal["GCJ-02"] | None = None

    @model_validator(mode="after")
    def _coordinates(self) -> "PublicMapEndpoint":
        if not self.name and not self.location:
            raise ValueError("endpoint needs a name or coordinates")
        if self.location is not None:
            if (
                normalize_location(self.location) is None
                or self.coordinate_system != "GCJ-02"
            ):
                raise ValueError("invalid GCJ-02 endpoint")
        elif self.coordinate_system is not None:
            raise ValueError("coordinate system requires coordinates")
        return self


class PublicMapDetails(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    category: CardText | None = None
    city: CardText | None = None
    district: CardText | None = None
    opening_hours: CardText | None = None
    steps: tuple[CardText, ...] | None = Field(default=None, max_length=12)


class _MapCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: SafeIdentifier
    title: CardText
    source_id: SafeIdentifier
    url: str | None = None
    distance_m: Metric | None = None
    details: PublicMapDetails | None = None

    @field_validator("url")
    @classmethod
    def _safe_link(cls, value: str | None) -> str | None:
        if value is not None and not is_safe_amap_url(value):
            raise ValueError("map link is not an approved HTTPS URI")
        return value


class PublicPlaceCard(_MapCard):
    kind: Literal["place"]
    poi_id: PoiIdentifier | None = None
    address: CardText | None = None
    location: str | None = None
    coordinate_system: Literal["GCJ-02"] | None = None

    @model_validator(mode="after")
    def _coordinates(self) -> "PublicPlaceCard":
        if self.location is not None:
            if (
                normalize_location(self.location) is None
                or self.coordinate_system != "GCJ-02"
            ):
                raise ValueError("invalid GCJ-02 place")
        elif self.coordinate_system is not None:
            raise ValueError("coordinate system requires coordinates")
        if self.url:
            url = urlsplit(self.url)
            params = parse_qs(url.query)
            location = params.get("position", [None])[0]
            if (
                url.path != "/marker"
                or ("poiid" in params and params["poiid"][0] != self.poi_id)
                or (
                    "poiid" not in params
                    and normalize_location(location)
                    != normalize_location(self.location)
                )
            ):
                raise ValueError("map link must refer to the card location")
        return self


class PublicRouteCard(_MapCard):
    kind: Literal["route"]
    origin: PublicMapEndpoint | None = None
    destination: PublicMapEndpoint | None = None
    mode: Literal["driving", "walking", "bicycling", "transit"] | None = None
    duration_s: Metric | None = None

    @model_validator(mode="after")
    def _matching_link(self) -> "PublicRouteCard":
        if self.url:
            url = urlsplit(self.url)
            params = parse_qs(url.query)
            if url.path != "/navigation" or not self.origin or not self.destination:
                raise ValueError("navigation requires both endpoints")
            for key, endpoint in (("from", self.origin), ("to", self.destination)):
                if normalize_location(params.get(key, [None])[0]) != normalize_location(
                    endpoint.location
                ):
                    raise ValueError("navigation coordinates disagree with endpoints")
            if params.get("mode", [None])[0] != {
                "driving": "car",
                "walking": "walk",
                "bicycling": "ride",
                "transit": "bus",
            }.get(self.mode or ""):
                raise ValueError("navigation mode disagrees with route")
        return self


PublicMapCard = Annotated[
    PublicPlaceCard | PublicRouteCard, Field(discriminator="kind")
]


class PublicAnswerSegment(BaseModel):
    """One audited display segment; no provider or tool fields are accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["content", "heading", "separator", "references"]
    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
    ]
    evidence_ids: tuple[SafeIdentifier, ...] = Field(default=(), max_length=32)


class PublicAnswer(BaseModel):
    """The only answer representation permitted across the public API boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PublicTerminalStatus | None = None
    audited: Literal[True] | None = None
    segments: tuple[PublicAnswerSegment, ...] = Field(default=(), max_length=64)
    evidence_parent_ids: tuple[SafeIdentifier, ...] = Field(default=(), max_length=256)
    route: PublicRoute | None = None
    runtime_config_snapshot_id: SafeIdentifier | None = None
    client_provenance: PublicClientProvenance | None = None
    citation_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    tool_audited: Literal[True] | None = None
    external_sources: tuple[PublicExternalSource, ...] | None = Field(
        default=None, max_length=16
    )
    cards: tuple[PublicMapCard, ...] | None = Field(default=None, max_length=32)

    @model_serializer(mode="wrap")
    def _preserve_legacy_shape(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        serialized = handler(self)
        for key in ("tool_audited", "external_sources", "cards"):
            if serialized.get(key) is None:
                serialized.pop(key, None)
        return serialized

    @model_validator(mode="after")
    def _audited_segments_only(self) -> "PublicAnswer":
        has_document_evidence = bool(self.evidence_parent_ids) or any(
            s.evidence_ids for s in self.segments
        )
        if (self.cards or self.external_sources) and self.tool_audited is not True:
            raise ValueError(
                "tool results require a separate source audit, including in chat"
            )
        if self.tool_audited is True and not self.external_sources:
            raise ValueError("tool audit requires provenance")
        sources = {source.id: source for source in self.external_sources or ()}
        if len(sources) != len(self.external_sources or ()):
            raise ValueError("duplicate source identifiers")
        if len({c.id for c in self.cards or ()}) != len(self.cards or ()):
            raise ValueError("duplicate card identifiers")
        if any(
            c.source_id not in sources or sources[c.source_id].provider != "高德地图"
            for c in self.cards or ()
        ):
            raise ValueError("every map card must reference a map source")
        if has_document_evidence and self.audited is not True:
            raise ValueError("tool audit cannot replace the document evidence audit")
        if self.route == "chat" and not has_document_evidence:
            if (
                self.audited is not None
                or self.evidence_parent_ids
                or self.citation_coverage is not None
            ):
                raise ValueError("chat must not claim document audits or citations")
            if any(s.evidence_ids or s.kind != "content" for s in self.segments):
                raise ValueError("chat permits only uncited content")
        elif (
            self.segments and self.audited is not True and self.tool_audited is not True
        ):
            raise ValueError("public answer segments must be audited")
        if self.status is None and not self.segments:
            raise ValueError(
                "public answer must contain audited segments or a safe status"
            )
        return self


def project_public_answer(
    value: object,
    *,
    evidence_parent_ids: Iterable[str] | None = None,
    route: str | None = None,
    runtime_config_snapshot_id: str | None = None,
    require_audited: bool = False,
) -> PublicAnswer | None:
    """Allowlist and strictly validate an untrusted stored or graph answer.

    Unknown fields are deliberately discarded before strict validation.  The
    strict model remains useful at typed call sites where unknown fields must
    be rejected, while persistence/API boundaries can safely salvage reviewed
    fields without ever returning raw provider or tool material.
    """
    if not isinstance(value, Mapping):
        return None
    candidate: dict[str, object] = {}
    for field_name in (
        "status",
        "audited",
        "route",
        "runtime_config_snapshot_id",
        "client_provenance",
        "citation_coverage",
        "tool_audited",
    ):
        if field_name in value:
            candidate[field_name] = value[field_name]

    for field_name, allowed in (
        ("external_sources", ("id", "tool_id", "provider", "observed_at")),
        (
            "cards",
            (
                "id",
                "kind",
                "title",
                "source_id",
                "poi_id",
                "url",
                "address",
                "location",
                "coordinate_system",
                "origin",
                "destination",
                "mode",
                "distance_m",
                "duration_s",
                "details",
            ),
        ),
    ):
        if field_name not in value:
            continue
        items = value[field_name]
        if items is None:
            continue
        if not isinstance(items, (list, tuple)) or any(
            not isinstance(item, Mapping) for item in items
        ):
            return None
        cleaned = [{k: item[k] for k in allowed if k in item} for item in items]
        if field_name == "cards":
            for card in cleaned:
                for nested, keys in (
                    ("origin", ("name", "location", "coordinate_system")),
                    ("destination", ("name", "location", "coordinate_system")),
                    (
                        "details",
                        ("category", "city", "district", "opening_hours", "steps"),
                    ),
                ):
                    if isinstance(card.get(nested), Mapping):
                        card[nested] = {
                            k: card[nested][k] for k in keys if k in card[nested]
                        }
        candidate[field_name] = cleaned

    segments = value.get("segments")
    if isinstance(segments, (list, tuple)):
        candidate["segments"] = [
            {
                field_name: item[field_name]
                for field_name in ("kind", "text", "evidence_ids")
                if field_name in item
            }
            for item in segments
            if isinstance(item, Mapping)
        ]

    if evidence_parent_ids is None:
        stored_parent_ids = value.get("evidence_parent_ids")
        if isinstance(stored_parent_ids, (list, tuple)):
            candidate["evidence_parent_ids"] = list(stored_parent_ids)
    else:
        candidate["evidence_parent_ids"] = list(dict.fromkeys(evidence_parent_ids))
    if route is not None:
        candidate["route"] = route
    if runtime_config_snapshot_id is not None:
        candidate["runtime_config_snapshot_id"] = runtime_config_snapshot_id

    try:
        projected = PublicAnswer.model_validate(candidate)
    except (ValidationError, TypeError, ValueError):
        return None
    if require_audited and (
        (projected.audited is not True and projected.tool_audited is not True)
        or not projected.segments
    ):
        return None
    return projected
