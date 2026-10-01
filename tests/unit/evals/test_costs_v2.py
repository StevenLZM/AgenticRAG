import pytest

from evals.costs_v2 import Pricing, price_requests, provider_requests


def pricing():
    return Pricing.model_validate({"currency": "CNY", "effective_date": "2026-10-01", "source": "operator-pinned-tariff",
        "models": {"m": {"input_per_million": 2, "cached_input_per_million": .2, "output_per_million": 3}}})


def test_price_actual_usage_including_cache_and_reject_unknown():
    request = {"status": "completed", "model": "m", "input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 100}
    assert price_requests([request], pricing())["value"] == pytest.approx(.00122)
    unknown = price_requests([request, {"status": "running"}], pricing())
    assert unknown["value"] is None and unknown["known_subtotal"] == pytest.approx(.00122)
    assert price_requests([{**request, "model": "unpriced"}], pricing())["value"] is None


def test_incomplete_request_events_do_not_claim_complete_cost():
    start = {"event_type": "PROVIDER_REQUEST_STARTED", "attributes": {"provider_request_id": "r", "requested_model": "m", "operation": "llm"}}
    finished = {"event_type": "PROVIDER_REQUEST_COMPLETED", "attributes": {**start["attributes"], "actual_model": "m", "input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 0}}
    assert provider_requests([start, finished])["status"] == "available"
    assert provider_requests([finished])["status"] == "incomplete"
    assert provider_requests([start])["status"] == "incomplete"
    missing_attempt = {"event_type": "LLM_COMPLETED", "attributes": {"attempts": 2}}
    assert provider_requests([start, finished, missing_attempt])["status"] == "incomplete"
