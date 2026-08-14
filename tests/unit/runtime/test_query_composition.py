"""Production Query dependency composition contracts."""

from __future__ import annotations

import pytest

from agentic_rag.config import Settings
from agentic_rag.runtime.query_composition import (
    QueryCompositionError,
    build_query_dependencies,
)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "mysql_dsn": "mysql+asyncmy://user:password@127.0.0.1:3306/app",
        "deepseek_base_url": "https://api.deepseek.com",
        "qwen_embedding_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.asyncio
async def test_query_composition_fails_closed_without_model_credentials() -> None:
    with pytest.raises(QueryCompositionError, match="credentials"):
        await build_query_dependencies(
            object(),
            _settings(
                deepseek_api_key="replace-with-key",
                qwen_api_key="replace-with-key",
            ),
        )


@pytest.mark.asyncio
async def test_query_composition_reports_missing_reranker_dependency() -> None:
    with pytest.raises(QueryCompositionError, match="reranker"):
        await build_query_dependencies(
            object(),
            _settings(
                deepseek_api_key="deepseek-test-key",
                qwen_api_key="qwen-test-key",
            ),
        )
