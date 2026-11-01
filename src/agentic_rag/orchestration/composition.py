"""Role-specific deployment composition. New workers never open SQLite."""
from typing import Any

from openai import AsyncOpenAI

from agentic_rag.config import Settings
from agentic_rag.orchestration.evidence import EvidencePreparer
from agentic_rag.orchestration.execution import StageExecutor
from agentic_rag.orchestration.store import CoordinationStore
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
from agentic_rag.retrieval.adapters.elasticsearch import ElasticsearchBm25Index, ElasticsearchVectorIndex
from agentic_rag.retrieval.graph import RetrievalDependencies, RetrievalService
from agentic_rag.retrieval.parents import ParentFetcher
from agentic_rag.retrieval.reranker import Reranker
from agentic_rag.runtime.model_gateway import ModelGateway
from agentic_rag.runtime.query_composition import _credential, _OpenAIEmbedding, _SessionParentRepository


async def build_executor(container: Any, settings: Settings, role: str, *, retrieval_index: str | None = None) -> tuple[StageExecutor, list[Any]]:
    store = CoordinationStore(container.repositories.session_factory, context_sources=getattr(container, "context_sources", None), query_compiler=getattr(container, "query_compiler", None), capacity_limits={
        "model": settings.max_concurrent_llm_calls, "rerank": settings.max_concurrent_reranks,
        "tenant_model": settings.supervisor_tenant_model_limit,
    })
    if settings.expected_representation_version and store.context_sources is None:
        raise ValueError("enabled representation requires current SQL source authority")
    if settings.query_context_enabled and store.query_compiler is None:
        raise ValueError("enabled query context requires configured compiler")
    clients = []
    try:
        client = AsyncOpenAI(api_key=_credential(settings.deepseek_api_key, "DEEPSEEK_API_KEY"), base_url=settings.deepseek_base_url,
                             timeout=settings.query_run_timeout_seconds + 5, max_retries=0)
        clients.append(client)
        gateway = ModelGateway(client, max_retries=0)
        parents = ParentFetcher(_SessionParentRepository(container.repositories.session_factory))
        retrieval = None
        if role in ("all", "retrieval"):
            await ElasticsearchChildIndexStore(
                container.elasticsearch, embedding_model=settings.embedding_model,
                lexical_analysis=settings.lexical_analysis, expected_representation_version=settings.expected_representation_version,
            ).validate_search_target(settings.index_generation, index=retrieval_index)
            from sentence_transformers import CrossEncoder
            embedding_client = AsyncOpenAI(api_key=_credential(settings.qwen_api_key, "QWEN_API_KEY"), base_url=settings.qwen_embedding_base_url, timeout=30, max_retries=0)
            clients.append(embedding_client)
            from agentic_rag.runtime.context_composition import pair_adapter
            encoder = CrossEncoder(settings.reranker_model)
            retrieval = RetrievalService(RetrievalDependencies(
                embedding=_OpenAIEmbedding(embedding_client, settings.embedding_model, settings.embedding_dimensions),
                vector=ElasticsearchVectorIndex(container.elasticsearch, index_generation=settings.index_generation, index=retrieval_index, expected_representation_version=settings.expected_representation_version),
                lexical=ElasticsearchBm25Index(container.elasticsearch, index_generation=settings.index_generation, lexical_analysis=settings.lexical_analysis, expected_representation_version=settings.expected_representation_version, index=retrieval_index),
                reranker=Reranker(encoder, pair_tokenizer=pair_adapter(encoder, settings), model_version=settings.reranker_model, max_concurrent_reranks=settings.max_concurrent_reranks, max_candidates=200),
                parent_fetcher=parents, lane_timeout_seconds=10, context_sources=getattr(container, "context_sources", None), query_compiler=getattr(container, "query_compiler", None),
            ))
        return StageExecutor(store, gateway, retrieval=retrieval, preparer=EvidencePreparer(parents, context_sources=getattr(container, "context_sources", None))), clients
    except BaseException:
        for client in clients:
            await client.close()
        raise
