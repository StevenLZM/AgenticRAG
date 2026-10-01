"""Production composition root for the Query Worker graph."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from importlib import import_module
from typing import Any, cast

from openai import AsyncOpenAI

from agentic_rag.config import Settings
from agentic_rag.persistence.repositories import ParentChunk, ParentRepository
from agentic_rag.query.audit import (
    CitationValidator,
    EvidenceGrader,
    FaithfulnessAuditor,
    ParentRepositoryAuthorizationResolver,
)
from agentic_rag.query.evidence_builder import EvidenceBuilder, PackedEvidence
from agentic_rag.query.generation import AnswerGenerator
from agentic_rag.query.graph import QueryGraphDependencies
from agentic_rag.query.routing_policy import RuntimeCapabilities
from agentic_rag.persistence.conversations import SqlAlchemyConversationReader
from agentic_rag.query.research_loop import ResearchAgentLoop, ResearchLoopDependencies
from agentic_rag.query.subagents import (
    ChildResearchState,
    ChildWorker,
    SubagentDispatcher,
    SubagentTools,
)
from agentic_rag.query.tools import ResearchContext, ResearchToolset, RetrievalPort
from agentic_rag.retrieval.models import EvidenceBatch
from agentic_rag.retrieval.adapters.elasticsearch import (
    ElasticsearchBm25Index,
    ElasticsearchVectorIndex,
)
from agentic_rag.retrieval.graph import RetrievalDependencies, RetrievalService
from agentic_rag.retrieval.parents import ParentFetcher
from agentic_rag.retrieval.reranker import Reranker
from agentic_rag.runtime.model_gateway import ModelGateway, prompt_hashes
from agentic_rag.runtime.concurrency import ConcurrencyManager
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from agentic_rag.domain.models import UserScope
from agentic_rag.observability.logging import AgentEventEmitter
from agentic_rag.observability.tracing import TraceRecorder


class QueryCompositionError(RuntimeError):
    """Raised when the Query Worker cannot be safely composed."""


_PROMPTS = (
    "chat_v2",
    "router_v2",
    "evidence_grader_v2",
    "research_agent_v1",
    "generator_v1",
    "faithfulness_v1",
    "memory_extractor_v1",
    "context_compactor_v1",
)


class _OpenAIEmbedding:
    """Qwen-compatible embedding adapter owned by the Query Worker."""

    def __init__(self, client: AsyncOpenAI, model: str, dimensions: int) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            dimensions=self._dimensions,
        )
        vectors = [list(item.embedding) for item in response.data]
        if any(len(vector) != self._dimensions for vector in vectors):
            raise QueryCompositionError("embedding provider returned an invalid dimension")
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class _SessionParentRepository(ParentRepository):
    """Open a short scoped SQL session per parent hydration operation."""

    def __init__(self, session_factory: Any) -> None:
        self._factory = session_factory

    async def get_many(self, parent_ids: list[str], scope: Any) -> list[ParentChunk]:
        from agentic_rag.persistence.repositories import SqlAlchemyParentRepository

        async with self._factory() as session:
            return await SqlAlchemyParentRepository(session).get_many(parent_ids, scope)


def _credential(value: object, name: str) -> str:
    secret = value.get_secret_value() if value is not None and hasattr(value, "get_secret_value") else ""
    if not isinstance(secret, str) or not secret.strip() or secret.startswith("replace-with-"):
        raise QueryCompositionError(f"{name} credentials are required for Query Worker")
    return secret


def build_query_snapshot(settings: Settings) -> RuntimeConfigSnapshot:
    try:
        hashes = prompt_hashes(_PROMPTS)
    except (OSError, ValueError) as error:
        raise QueryCompositionError("query prompt artifacts are unavailable") from error
    if settings.embedding_dimensions != 1024:
        raise QueryCompositionError("Query Worker requires 1024-dimensional embeddings")
    return RuntimeConfigSnapshot(
        app_version="0.1.0",
        graph_version="query-v2",
        prompt_version="prompt-v2",
        routing_policy_version="routing-v2",
        prompt_hashes=tuple(sorted(hashes.items())),
        main_model_id=settings.main_model,
        light_model_id=settings.light_model,
        deepseek_protocol=settings.deepseek_protocol,
        embedding_model=settings.embedding_model,
        embedding_dimensions=1024,
        reranker_version=settings.reranker_model,
        retrieval_config_version="retrieval-v1",
        index_generation=settings.index_generation,
        memory_config_version="agent_memories_v1-v1",
        max_research_rounds=settings.max_research_rounds,
        max_answer_revisions=settings.max_answer_revisions,
        query_run_timeout_seconds=settings.query_run_timeout_seconds,
        max_evidence_tokens=settings.max_evidence_tokens,
        research_context_soft_limit_tokens=settings.research_context_soft_limit_tokens,
        max_parallel_subagents_per_run=settings.max_parallel_subagents_per_run,
    )


def build_subagent_dispatcher(
    *,
    retrieval: RetrievalPort,
    evidence_builder: EvidenceBuilder,
    snapshot: RuntimeConfigSnapshot,
    concurrency: ConcurrencyManager,
) -> SubagentDispatcher:
    """Compose bounded children that inherit only server-owned query context."""
    tools = ResearchToolset(retrieval, evidence_builder)

    async def child_worker(
        child: ChildResearchState,
        child_tools: SubagentTools,
    ) -> tuple[EvidenceBatch, PackedEvidence]:
        child_scope = UserScope.model_validate(dict(child.scope))
        context = ResearchContext(scope=child_scope, snapshot=snapshot)
        return await child_tools.retrieve_evidence(
            query=child.question,
            context=context,
            target_id=child.todo_id,
        )

    return SubagentDispatcher(
        tools=tools,
        concurrency=concurrency,
        worker=cast(ChildWorker, child_worker),
        memory_summary="",
        evidence_manifest={},
    )


async def build_query_dependencies(
    container: object,
    settings: Settings,
    *,
    child_index: str | None = None,
    concurrency: ConcurrencyManager | None = None,
) -> QueryGraphDependencies:
    """Build all process-owned QueryGraph collaborators from one snapshot."""
    deepseek_key = _credential(settings.deepseek_api_key, "AGENTIC_RAG_DEEPSEEK_API_KEY")
    qwen_key = _credential(settings.qwen_api_key, "AGENTIC_RAG_QWEN_API_KEY")
    try:
        cross_encoder_type = getattr(import_module("sentence_transformers"), "CrossEncoder")
    except ImportError as error:
        raise QueryCompositionError(
            "reranker dependency sentence-transformers is not installed"
        ) from error

    snapshot = build_query_snapshot(settings)
    shared_concurrency = concurrency or ConcurrencyManager(
        run_limit=settings.max_concurrent_query_runs,
        llm_limit=settings.max_concurrent_llm_calls,
        reranker_limit=settings.max_concurrent_reranks,
        per_run_subagent_limit=snapshot.max_parallel_subagents_per_run,
    )
    elasticsearch = getattr(container, "elasticsearch", None)
    repositories = getattr(container, "repositories", None)
    artifacts = getattr(container, "artifacts", None)
    event_repository = getattr(container, "event_repository", None)
    memory = getattr(container, "memory_service", None)
    if elasticsearch is None or repositories is None or artifacts is None or event_repository is None:
        raise QueryCompositionError("Query Worker container is missing persistence boundaries")
    if memory is None:
        raise QueryCompositionError("Query Worker container is missing memory boundary")

    # The SDK transport must outlive the gateway's per-operation deadline so
    # timeout classification and the single retry budget stay in our boundary.
    deepseek_timeout = max(float(snapshot.query_run_timeout_seconds) + 5.0, 5.0)
    deepseek = AsyncOpenAI(
        api_key=deepseek_key,
        base_url=settings.deepseek_base_url,
        timeout=deepseek_timeout,
        max_retries=0,
    )
    qwen = AsyncOpenAI(api_key=qwen_key, base_url=settings.qwen_embedding_base_url, timeout=30.0, max_retries=0)
    try:
        # ModelGateway derives the diagnostic client timeout from the configured
        # provider client when the SDK exposes it.  Keep construction positional
        # for lightweight test doubles and alternate gateway adapters.
        gateway = ModelGateway(deepseek)
        embedding = _OpenAIEmbedding(qwen, settings.embedding_model, settings.embedding_dimensions)
        cross_encoder = cross_encoder_type(settings.reranker_model)
        reranker = Reranker(
            cross_encoder,
            model_version=settings.reranker_model,
            max_concurrent_reranks=settings.max_concurrent_reranks,
        )
        parents = _SessionParentRepository(repositories.session_factory)
        retrieval = RetrievalService(
            RetrievalDependencies(
                embedding=embedding,
                vector=ElasticsearchVectorIndex(
                    elasticsearch,
                    index_generation=settings.index_generation,
                    index=child_index,
                ),
                lexical=ElasticsearchBm25Index(
                    elasticsearch,
                    index_generation=settings.index_generation,
                    index=child_index,
                ),
                reranker=reranker,
                parent_fetcher=ParentFetcher(parents),
                lane_timeout_seconds=10.0,
            )
        )
        evidence_builder = EvidenceBuilder()
        subagents = build_subagent_dispatcher(
            retrieval=retrieval,
            evidence_builder=evidence_builder,
            snapshot=snapshot,
            concurrency=shared_concurrency,
        )
        research_loop = ResearchAgentLoop(
            ResearchLoopDependencies(
                gateway=gateway,
                retrieval=retrieval,
                evidence_builder=evidence_builder,
                subagents=subagents,
            )
        )
        emitter = AgentEventEmitter(
            event_repository,
            artifacts,
            runtime_config_snapshot_id=snapshot.snapshot_id,
        )
        dependencies = QueryGraphDependencies(
            memory=cast(Any, memory),
            gateway=gateway,
            retrieval=retrieval,
            evidence_builder=evidence_builder,
            evidence_grader=cast(Any, EvidenceGrader(gateway)),
            research_loop=research_loop,
            generator=AnswerGenerator(gateway),
            faithfulness_auditor=FaithfulnessAuditor(gateway),
            citation_validator=CitationValidator(),
            authorization_resolver=ParentRepositoryAuthorizationResolver(parents),
            event_repository=event_repository,
            trace_recorder=TraceRecorder(runtime_config_snapshot_id=snapshot.snapshot_id),
            event_emitter=emitter,
            concurrency=shared_concurrency,
            owned_resources=(deepseek, qwen, reranker),
            capabilities=RuntimeCapabilities(knowledge_base=True),
            conversations=SqlAlchemyConversationReader(repositories.session_factory),
        )
        return dependencies
    except QueryCompositionError:
        await deepseek.close()
        await qwen.close()
        raise
    except Exception as error:
        await deepseek.close()
        await qwen.close()
        raise QueryCompositionError(f"Query Worker dependency composition failed: {type(error).__name__}") from error


async def close_query_dependencies(dependencies: QueryGraphDependencies) -> None:
    """Close only resources owned by the Query composition root."""
    for resource in reversed(dependencies.owned_resources):
        close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if callable(close):
            value = close()
            if asyncio.iscoroutine(value):
                await value


__all__ = [
    "QueryCompositionError",
    "build_query_snapshot",
    "build_subagent_dispatcher",
    "build_query_dependencies",
    "close_query_dependencies",
]
