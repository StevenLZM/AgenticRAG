"""Opt-in live provider smoke checks; credentials are required when selected."""

from __future__ import annotations

import pytest

from agentic_rag.config import Settings
from scripts.live_model_smoke import run_live_model_smoke


pytestmark = pytest.mark.live_model


async def test_configured_live_models_record_ids_and_1024_dimension_embedding() -> None:
    settings = Settings()

    result = await run_live_model_smoke(settings)

    assert result.routing_model_id
    assert result.completion_model_id
    assert result.embedding_model_id
    assert result.embedding_dimensions == 1024
