"""Opt-in production dependency composition; no implicit model/store doubles."""

import hashlib
from agentic_rag.retrieval.context_evidence import ContextSourceHydrator
from agentic_rag.retrieval.query_compiler import QueryCompiler
from agentic_rag.persistence.context_registry import RegistrySourceAuthority
from agentic_rag.persistence.context_release import ReleaseStore


def validate_context_configuration(settings):
    if settings.context_release_enabled:
        roles = set(settings.context_deployment_roles)
        required = {"api", "ingestion", "rebuild"}
        if settings.supervisor_enabled:
            required.add("coordinator")
            if "all" not in roles:
                required.update(
                    ("supervisor", "retrieval", "analysis", "evidence", "answer")
                )
        else:
            required.add("query")
        if not required <= roles:
            raise ValueError(
                "release deployment inventory omits configured runtime roles"
            )
    if (
        settings.context_pipeline_enabled
        or settings.expected_representation_version
        or settings.query_context_enabled
    ) and not settings.context_release_enabled:
        raise ValueError("context features require durable release admission")
    if settings.context_pipeline_enabled:
        build = settings.context_build
        if (
            not build.enabled
            or settings.expected_representation_version != "context-v1"
        ):
            raise ValueError(
                "context ingestion requires matching context-v1 generation policy"
            )
        if not build.embedding_tokenizer_id:
            raise ValueError(
                "context ingestion requires model-bound embedding tokenizer identity"
            )
        if build.extraction_enabled and (
            not build.extraction_tokenizer_id or not build.verification_tokenizer_id
        ):
            raise ValueError(
                "context models require independently bound tokenizer identities"
            )
    if settings.context_build.enabled and not settings.context_pipeline_enabled:
        raise ValueError("enabled context build requires context pipeline")


def build_context_sources(factory, artifacts, settings):
    if settings.expected_representation_version is None:
        return None
    if factory is None or artifacts is None:
        raise ValueError("context sources require SQL registry and durable artifacts")
    return ContextSourceHydrator(
        artifacts=artifacts, authority=RegistrySourceAuthority(factory, artifacts)
    )


def build_query_compiler(
    settings, artifacts, *, profiles=None, directory=None, coverage=None
):
    return (
        QueryCompiler(
            profiles=profiles,
            directory=directory,
            coverage=coverage,
            artifacts=artifacts,
        )
        if settings.query_context_enabled
        else None
    )


def pair_adapter(encoder, settings):
    if not (settings.expected_representation_version or settings.query_context_enabled):
        return None
    from agentic_rag.retrieval.reranker import TokenizerPairAdapter

    capacity = getattr(encoder, "max_length", None)
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None or not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "enabled context reranker requires actual loaded fast tokenizer"
        )
    model_limit = getattr(getattr(encoder, "model", None), "config", None)
    limits = [
        value
        for value in (
            capacity,
            getattr(tokenizer, "model_max_length", None),
            getattr(model_limit, "max_position_embeddings", None),
        )
        if isinstance(value, int) and 0 < value < 1_000_000
    ]
    if not limits:
        raise ValueError("enabled context reranker requires finite model pair capacity")
    return TokenizerPairAdapter(tokenizer, max_pair_tokens=min(limits))


def tokenizer_identity(tokenizer, model):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("context tokenizer requires serializable model tokenizer")
    return f"{model}:{hashlib.sha256(backend.to_str().encode()).hexdigest()}"


def build_contextualizer(container, settings, tokenizer):
    if not settings.context_pipeline_enabled:
        return None, None
    validate_context_configuration(settings)
    from agentic_rag.ingestion.contextualization.service import ContextualizationService
    from agentic_rag.persistence.context_builds import SqlAlchemyContextBuildStore
    from agentic_rag.persistence.context_limiter import (
        SqlContextModelLimiter,
        LimitedContextModel,
    )
    from agentic_rag.ingestion.contextualization.gateway import GatewayContextModel
    from agentic_rag.runtime.query_composition import build_query_snapshot, _credential

    loaded = getattr(tokenizer, "tokenizer", None)
    if (
        loaded is None
        or tokenizer_identity(loaded, settings.embedding_model)
        != settings.context_build.embedding_tokenizer_id
    ):
        raise ValueError(
            "embedding tokenizer does not match frozen context build identity"
        )
    model, client, counters = None, None, {}
    build = settings.context_build
    if build.extraction_enabled:
        from transformers import AutoTokenizer
        from openai import AsyncOpenAI

        for model_id, expected in (
            (build.extraction_model, build.extraction_tokenizer_id),
            (build.verification_model, build.verification_tokenizer_id),
        ):
            path = settings.context_model_tokenizers.get(model_id)
            if not path:
                raise ValueError("missing generation-model tokenizer binding")
            generation_tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True
            )
            if tokenizer_identity(generation_tokenizer, model_id) != expected:
                raise ValueError(
                    "generation tokenizer does not match frozen model identity"
                )
            counters[model_id] = generation_tokenizer
        snapshot = build_query_snapshot(settings)
        if snapshot.provider_config_fingerprint != build.provider_fingerprint:
            raise ValueError(
                "context provider differs from frozen provider fingerprint"
            )
        client = AsyncOpenAI(
            api_key=_credential(settings.deepseek_api_key, "DEEPSEEK_API_KEY"),
            base_url=settings.deepseek_base_url,
            max_retries=0,
            timeout=build.timeout_seconds,
        )
        from agentic_rag.observability.provider_usage import install_provider_meter

        install_provider_meter(client)
        model = LimitedContextModel(
            GatewayContextModel(client=client, snapshot=snapshot, build=build),
            SqlContextModelLimiter(
                container.repositories.session_factory,
                limit=build.max_concurrency,
                timeout_seconds=build.timeout_seconds,
            ),
        )
    service = ContextualizationService(
        artifacts=container.artifacts,
        store=SqlAlchemyContextBuildStore(container.repositories.session_factory),
        model=model,
        count_tokens=lambda text: len(loaded.encode(text, add_special_tokens=False)),
        max_concurrency=build.max_concurrency,
        count_model_tokens=(
            lambda text, model_id: len(
                counters[model_id].encode(text, add_special_tokens=False)
            )
        )
        if counters
        else None,
    )
    return service, client


async def validate_context_role(container, role):
    if not container.settings.context_release_enabled:
        return
    if not container.settings.context_instance_id:
        raise ValueError("release-controlled role requires explicit instance identity")
    from sqlalchemy import inspect

    required = {
        "retrieval_builds",
        "retrieval_rebuild_jobs",
        "context_release_gate",
        "context_release_journal",
        "context_model_capacity",
        "context_build_budgets",
        "context_build_items",
        "context_build_reservations",
        "context_build_observations",
        "context_maintenance_nonces",
    }
    async with container.mysql_engine.connect() as connection:
        names = await connection.run_sync(
            lambda conn: set(inspect(conn).get_table_names())
        )
    if not required <= names:
        raise ValueError(
            "context schema missing; apply additive migrations in maintenance first"
        )
    async with container.mysql_engine.connect() as connection:
        columns = await connection.run_sync(
            lambda conn: {
                name: {column["name"] for column in inspect(conn).get_columns(name)}
                for name in ("context_release_gate", "context_release_journal")
            }
        )
    if (
        not {"active_journal_id", "execution_token", "execution_effect"}
        <= columns["context_release_gate"]
        or "recovery_receipts" not in columns["context_release_journal"]
    ):
        raise ValueError("context execution fence schema missing; apply migration 0012")
    from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore

    await ElasticsearchChildIndexStore(
        container.elasticsearch,
        embedding_model=container.settings.embedding_model,
        lexical_analysis=container.settings.lexical_analysis,
        expected_representation_version=container.settings.expected_representation_version,
    ).validate_search_target(container.settings.index_generation)
    await ReleaseStore(container.repositories.session_factory).attest(
        role, container.runtime_snapshot.snapshot_id
    )
