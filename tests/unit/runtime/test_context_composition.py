import pytest
from agentic_rag.config import Settings as SettingsModel
from agentic_rag.runtime.query_composition import build_query_snapshot
from agentic_rag.runtime.context_composition import (
    validate_context_configuration,
    build_context_sources,
)


def Settings(**kwargs):
    return SettingsModel(
        mysql_dsn="sqlite+aiosqlite:///:memory:",
        deepseek_base_url="http://localhost/v1",
        qwen_embedding_base_url="http://localhost/v1",
        **kwargs,
    )


def test_disabled_composition_needs_no_context_dependencies():
    settings = Settings()
    validate_context_configuration(settings)
    assert build_context_sources(None, None, settings) is None
    assert build_query_snapshot(settings).expected_representation_version is None


def test_context_read_policy_is_explicit_and_frozen():
    base = Settings()
    enabled = Settings(
        expected_representation_version="context-v1", context_release_enabled=True
    )
    validate_context_configuration(enabled)
    snapshot = build_query_snapshot(enabled)
    assert snapshot.expected_representation_version == "context-v1"
    assert snapshot.snapshot_id != build_query_snapshot(base).snapshot_id
    with pytest.raises(ValueError, match="release"):
        validate_context_configuration(
            Settings(expected_representation_version="context-v1")
        )


def test_structural_ingestion_requires_real_tokenizer_but_no_llm_budget():
    from agentic_rag.runtime.models import ContextBuildConfig

    settings = Settings(
        context_pipeline_enabled=True,
        expected_representation_version="context-v1",
        context_release_enabled=True,
        context_build=ContextBuildConfig(enabled=True),
    )
    with pytest.raises(ValueError, match="tokenizer"):
        validate_context_configuration(settings)


def test_structural_service_uses_sql_store_and_actual_embedding_tokenizer(tmp_path):
    from types import SimpleNamespace
    from agentic_rag.persistence.artifacts import LocalArtifactStore
    from agentic_rag.runtime.models import ContextBuildConfig
    from agentic_rag.runtime.context_composition import (
        build_contextualizer,
        tokenizer_identity,
    )
    from agentic_rag.persistence.context_builds import SqlAlchemyContextBuildStore

    class Tokenizer:
        backend_tokenizer = SimpleNamespace(to_str=lambda: '{"vocab":"offline"}')

        def encode(self, text, **kwargs):
            return list(text)

    raw = Tokenizer()
    settings = Settings(
        context_pipeline_enabled=True,
        expected_representation_version="context-v1",
        context_release_enabled=True,
        context_build=ContextBuildConfig(
            enabled=True,
            embedding_tokenizer_id=tokenizer_identity(raw, "text-embedding-v3"),
        ),
    )
    container = SimpleNamespace(
        artifacts=LocalArtifactStore(tmp_path),
        repositories=SimpleNamespace(session_factory=object()),
    )
    service, client = build_contextualizer(
        container, settings, SimpleNamespace(tokenizer=raw)
    )
    assert isinstance(service._store, SqlAlchemyContextBuildStore)
    assert service._model is None and client is None and service._count("正文") == 2
    with pytest.raises(ValueError, match="identity"):
        build_contextualizer(
            container,
            settings.model_copy(update={"embedding_model": "other"}),
            SimpleNamespace(tokenizer=raw),
        )


def test_pair_adapter_uses_loaded_encoder_and_smaller_actual_capacity():
    from types import SimpleNamespace
    from agentic_rag.runtime.context_composition import pair_adapter

    tokenizer = SimpleNamespace(model_max_length=512, is_fast=True)
    encoder = SimpleNamespace(
        max_length=384,
        tokenizer=tokenizer,
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024)),
    )
    adapter = pair_adapter(
        encoder,
        Settings(
            expected_representation_version="context-v1", context_release_enabled=True
        ),
    )
    assert adapter._tokenizer is tokenizer and adapter.max_pair_tokens == 384
    tokenizer.is_fast = False
    with pytest.raises(ValueError, match="fast tokenizer"):
        pair_adapter(encoder, Settings(expected_representation_version="context-v1", context_release_enabled=True))


@pytest.mark.asyncio
async def test_frozen_context_policy_rejects_legacy_hit_and_citation_markers(tmp_path):
    from agentic_rag.retrieval.context_evidence import ContextSourceHydrator
    from agentic_rag.persistence.artifacts import LocalArtifactStore
    from tests.unit.query.test_evidence_builder import child, SCOPE, SNAPSHOT
    from agentic_rag.query.audit import ParentRepositoryAuthorizationResolver
    from agentic_rag.query.evidence_builder import EvidenceManifestEntry
    from agentic_rag.orchestration.authorization import (
        authorize_sources,
        SourceAccessRevoked,
    )

    snapshot = SNAPSHOT.model_copy(
        update={"expected_representation_version": "context-v1"}
    )
    hydrator = ContextSourceHydrator(
        artifacts=LocalArtifactStore(tmp_path), authority=object()
    )
    with pytest.raises(ValueError, match="legacy hit"):
        await hydrator.hydrate([child("parent")], SCOPE, snapshot, None)

    class Parents:
        async def get_many(self, ids, scope):
            return []

    entry = EvidenceManifestEntry(
        evidence_id="E1",
        parent_id="p",
        document_id="d",
        document_version_id="v",
        heading_path=(),
        ast_locator="x",
    )
    assert await ParentRepositoryAuthorizationResolver(Parents()).resolve(
        {"E1": entry}, SCOPE, snapshot
    ) == {"E1": {}}
    with pytest.raises(SourceAccessRevoked, match="context-v1"):
        await authorize_sources(
            None,
            {
                "data": {
                    "snapshot": snapshot.model_dump(mode="json"),
                    "enforce_source_access": False,
                }
            },
            [{"schema_version": 1}],
        )
