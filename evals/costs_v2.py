"""Price observed requests against an explicitly pinned tariff, not token guesses."""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field

from evals.gold_v2_models import FrozenModel, canonical_hash


class ModelRate(FrozenModel):
    input_per_million: float = Field(ge=0, allow_inf_nan=False)
    cached_input_per_million: float = Field(ge=0, allow_inf_nan=False)
    output_per_million: float = Field(ge=0, allow_inf_nan=False)


class Pricing(FrozenModel):
    currency: Literal["CNY", "USD"]
    effective_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    source: str = Field(min_length=1)
    basis: Literal["fixed_tariff_estimate"] = "fixed_tariff_estimate"
    models: dict[str, ModelRate] = Field(min_length=1)

    @property
    def fingerprint(self):
        return canonical_hash(self.model_dump(mode="json"))


def provider_requests(timeline):
    started, ended = {}, {}
    malformed = False
    for event in timeline:
        kind, attrs = event["event_type"], event.get("attributes", {})
        if not kind.startswith("PROVIDER_REQUEST_"):
            continue
        identity = attrs.get("provider_request_id")
        if not identity:
            malformed = True
            continue
        bucket = started if kind == "PROVIDER_REQUEST_STARTED" else ended
        if identity in bucket:
            malformed = True
        bucket[identity] = event
    requests = []
    for identity in sorted(started.keys() | ended.keys()):
        final = ended.get(identity, {})
        attrs = {**started.get(identity, {}).get("attributes", {}), **final.get("attributes", {})}
        requests.append({"request_id": identity, "kind": attrs.get("operation"),
            "status": "completed" if identity in started and final.get("event_type") == "PROVIDER_REQUEST_COMPLETED" else "unknown",
            "model": attrs.get("actual_model") or attrs.get("requested_model"),
            **{key: attrs.get(key) for key in ("input_tokens", "output_tokens", "cached_input_tokens")}})
    complete = bool(requests) and not malformed and all(r["status"] == "completed"
        and all(type(r[key]) is int and r[key] >= 0 for key in ("input_tokens", "output_tokens")) for r in requests)
    expected_llm_attempts = sum(e.get("attributes", {}).get("attempts", 1) for e in timeline
        if e["event_type"] == "LLM_COMPLETED" and type(e.get("attributes", {}).get("attempts", 1)) is int)
    if sum(r["kind"] == "llm" for r in requests) < expected_llm_attempts:
        complete = False
    return {"status": "available" if complete else "incomplete" if requests else "unknown",
            "requests": requests, "observed_request_count": len(requests),
            "minimum_llm_attempts_from_gateway": expected_llm_attempts}


def price_requests(requests, pricing: Pricing | None):
    subtotal, unknown = Decimal(0), 0
    for request in requests:
        rate = pricing.models.get(request.get("model")) if pricing else None
        input_tokens, output_tokens, cached = (request.get(k) for k in ("input_tokens", "output_tokens", "cached_input_tokens"))
        if rate and cached is None and rate.input_per_million == rate.cached_input_per_million:
            cached = 0
        if (request.get("status") != "completed" or rate is None
                or any(type(v) is not int or v < 0 for v in (input_tokens, output_tokens, cached))
                or cached > input_tokens):
            unknown += 1
            continue
        subtotal += ((input_tokens - cached) * Decimal(str(rate.input_per_million))
                     + cached * Decimal(str(rate.cached_input_per_million))
                     + output_tokens * Decimal(str(rate.output_per_million))) / 1_000_000
    known = pricing is not None and bool(requests) and unknown == 0
    return {"value": float(subtotal) if known else None, "known_subtotal": float(subtotal), "unknown_requests": unknown,
            "requests": len(requests), "status": "available" if known else "unknown", "scope": "provider_api_only",
            "currency": pricing.currency if pricing else None, "pricing_fingerprint": pricing.fingerprint if pricing else None,
            "basis": "fixed_tariff_estimate_from_provider_usage"}
