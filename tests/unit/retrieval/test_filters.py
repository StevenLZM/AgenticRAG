"""Contract tests for server-owned retrieval filters."""

import pytest
from pydantic import ValidationError

from agentic_rag.domain.models import UserScope
from agentic_rag.retrieval.filters import FilterBuilder
from agentic_rag.retrieval.models import DateRange, RetrievalRequest
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="0.1.0",
    graph_version="graph-v1",
    prompt_version="prompt-v1",
    main_model_id="deepseek-v4-pro",
    light_model_id="deepseek-v4-flash",
    embedding_model="text-embedding-v3",
    embedding_dimensions=1024,
    reranker_version="bge-reranker-v2-m3",
    retrieval_config_version="retrieval-v1",
    index_generation="index-v1",
    memory_config_version="memory-v1",
)


def test_filter_builder_injects_user_active_and_generation() -> None:
    """Server-owned scope fields are injected into every search filter."""
    request = RetrievalRequest(query="合同期限", search_type="policy")

    result = FilterBuilder().build(request, UserScope(user_id="u1"), SNAPSHOT)

    assert result.user_id == "u1"
    assert result.is_active is True
    assert result.index_generation == SNAPSHOT.index_generation


def test_request_schema_rejects_user_id() -> None:
    """Agent requests cannot choose another user's data scope."""
    with pytest.raises(ValidationError):
        RetrievalRequest.model_validate({"query": "x", "user_id": "u2"})


@pytest.mark.parametrize("server_owned_field", ["user_id", "is_active", "index_generation"])
def test_request_schema_rejects_server_owned_filter_fields(
    server_owned_field: str,
) -> None:
    """Agent input cannot override any scope or index-generation boundary."""
    with pytest.raises(ValidationError):
        RetrievalRequest.model_validate({"query": "x", server_owned_field: "u2"})


def test_filter_builder_preserves_agent_selectors() -> None:
    """Safe query selectors survive server-side scope injection unchanged."""
    request = RetrievalRequest(
        query=" 合同期限 ",
        search_type="policy",
        document_ids=("doc-1",),
        content_types=("contract",),
        date_range=DateRange(start="2026-01-01", end="2026-01-31"),
    )

    result = FilterBuilder().build(request, UserScope(user_id="u1"), SNAPSHOT)

    assert result.search_type == "policy"
    assert result.document_ids == ("doc-1",)
    assert result.content_types == ("contract",)
    assert result.date_range == DateRange(start="2026-01-01", end="2026-01-31")


def test_date_range_rejects_an_end_before_its_start() -> None:
    """Invalid date boundaries cannot reach a retrieval adapter."""
    with pytest.raises(ValidationError, match="date range start must not exceed end"):
        DateRange(start="2026-02-01", end="2026-01-31")
