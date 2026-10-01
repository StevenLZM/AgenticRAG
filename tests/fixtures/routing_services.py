"""Opt-in isolated services, real upload/ingestion and production query wiring.

Reuses the real-query fixture's safety guards/lifecycle helpers, not its synthetic
seed, Mem0 write smoke, or injected degradation. No production aliases are changed.
"""
from contextlib import AsyncExitStack
from pathlib import Path
import asyncio
import hashlib

import httpx
import pytest
from alembic import command
from alembic.config import Config

from agentic_rag.api.app import create_app
from agentic_rag.bootstrap import build_container
from agentic_rag.domain.models import JobStatus, UserScope
from agentic_rag.runtime.run_manager import RunManager, TransactionalRunRepository
from agentic_rag.testing.isolated_query_broker import IsolatedQueryBroker
from tests.fixtures.query_services import (
    RealQueryRuntime, RealQueryFixture, _require_real_provider_services,
    _cleanup_real_provider_query_index, _cleanup_real_provider_mem0_index,
    _cleanup_real_provider_redis, _strict_container_close,
)
from scripts.real_acceptance_evidence import (
    create_isolated_mysql_database, drop_isolated_mysql_database, isolated_mysql_dsn,
    require_service_backup_admin_dsn,
)


DOCUMENTS = {
    "北京天气报告.txt": "北京历史天气报告（2023年7月）。平均气温28摄氏度，月降雨量150毫米。报告结论：雨季应准备雨具。本报告不是今天的实时天气。",
    "刘泽明简历.txt": "刘泽明简历。工作经历：京东，2020年1月至2022年1月，后端工程师。负责订单系统、库存服务及接口优化。项目经验：订单缓存优化，使用Python和Redis降低查询延迟。",
    "李四简历.txt": "李四简历。工作经历：腾讯，2021年1月至2024年1月，数据工程师。负责离线数据管道。项目经验：使用Spark构建用户行为分析平台。",
}


class RoutingRuntime(RealQueryRuntime):
    async def wait_for_terminal(self, run_id, *, restart_worker=False, timeout=180):
        # Parent live smoke injects a degradation event: never do that here.
        return await RealQueryFixture.wait_for_terminal(
            self, run_id, restart_worker=restart_worker, timeout=timeout)


async def _ingest_documents(container):
    # Import heavy parser/tokenizer only after explicit live-service opt-in.
    from scripts.run_ingestion_worker import (
        QwenEmbeddingAdapter, HuggingFaceTokenizer, CHILD_MAX_TOKENS, DocumentParser,
        GlobalAssembler, ChunkingPipeline, ContentSafetyScanner, DefaultUploadSafetyScanner,
        SqlAlchemyIngestionJobStore, SqlAlchemyParentStagingStore, ElasticsearchChildIndexStore,
        SqlAlchemyPublicationRepository, VersionPublisher, IndexWriter, DefaultIngestionPipeline,
        IngestionGraph,
    )
    settings = container.settings
    factory = container.repositories.session_factory
    jobs = SqlAlchemyIngestionJobStore(factory)
    parents = SqlAlchemyParentStagingStore(factory)
    children = ElasticsearchChildIndexStore(container.elasticsearch)
    embedding = QwenEmbeddingAdapter(settings)
    try:
        pipeline = DefaultIngestionPipeline(jobs=jobs, artifacts=container.artifacts,
            upload_scanner=DefaultUploadSafetyScanner(max_upload_bytes=settings.max_upload_bytes),
            parser=DocumentParser(source_loader=container.artifacts.read_bytes, artifacts=container.artifacts),
            assembler=GlobalAssembler(), content_scanner=ContentSafetyScanner(),
            chunker=ChunkingPipeline(HuggingFaceTokenizer.from_pretrained(
                settings.embedding_tokenizer_model, max_tokens=CHILD_MAX_TOKENS)),
            index_writer=IndexWriter(embedding=embedding, parent_store=parents, child_store=children, artifacts=container.artifacts),
            publisher=VersionPublisher(repository=SqlAlchemyPublicationRepository(factory),
                parent_store=parents, child_store=children, artifacts=container.artifacts))
        async with container.checkpoints.open_ingestion() as checkpoint:
            graph = IngestionGraph(pipeline=pipeline, checkpointer=checkpoint)
            for filename, content in DOCUMENTS.items():
                job = await container.document_service.create_upload(
                    UserScope(user_id=settings.default_user_id), filename, "text/plain", content.encode())
                claim = await jobs.claim(job.id, "routing-acceptance", lease_seconds=3600)
                assert claim is not None
                await graph.run(claim)
                assert await jobs.get_status(job.id) == JobStatus.COMPLETED
        await container.elasticsearch.indices.refresh(index=f"agenticrag-children-{settings.index_generation}")
    finally:
        await embedding.close()


@pytest.fixture
async def routing_runtime(tmp_path: Path):
    settings = _require_real_provider_services(tmp_path)
    admin_dsn = require_service_backup_admin_dsn()
    suffix = settings.index_generation.removeprefix("real-query-")
    database = f"agentic_rag_acceptance_source_{suffix}"
    async with AsyncExitStack() as stack:
        await asyncio.to_thread(create_isolated_mysql_database, admin_dsn, database)
        stack.push_async_callback(asyncio.to_thread, drop_isolated_mysql_database, admin_dsn, database)
        settings = settings.model_copy(update={"mysql_dsn": isolated_mysql_dsn(admin_dsn, database),
                                               "allow_evaluation_requests": True})
        migration = Config(str(Path("alembic.ini").resolve()))
        migration.set_main_option("sqlalchemy.url", settings.mysql_dsn)
        await asyncio.to_thread(command.upgrade, migration, "head")
        container = build_container(settings)
        stack.push_async_callback(_strict_container_close, container)
        stack.push_async_callback(_cleanup_real_provider_query_index, container, settings)
        stack.push_async_callback(_cleanup_real_provider_mem0_index, container, settings)
        broker = IsolatedQueryBroker(container.broker, namespace=settings.index_generation)
        stack.push_async_callback(_cleanup_real_provider_redis, container, broker)
        container.run_manager = RunManager(session_factory=container.repositories.session_factory,
            runs=TransactionalRunRepository(container.repositories.session_factory, outbox_stream_name=broker.query_stream))
        await _ingest_documents(container)
        app = create_app(settings, container=container)
        lifespan = app.router.lifespan_context(app)
        await stack.enter_async_context(lifespan)
        client = await stack.enter_async_context(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://routing-acceptance"))
        runtime = RoutingRuntime(settings=settings, container=container, app=app,
            client=client, app_lifespan=lifespan, parent_id="", broker=broker)
        # Close only process resources here; the stack owns isolated persistence.
        from agentic_rag.runtime.query_composition import close_query_dependencies
        async def close_worker():
            try:
                await runtime.stop_worker()
            finally:
                try:
                    await runtime._close_checkpoint()
                finally:
                    if runtime._dependencies is not None:
                        await close_query_dependencies(runtime._dependencies)
        stack.push_async_callback(close_worker)
        await runtime.start_worker()
        runtime.routing_evaluation = {"session_id": suffix, "memory_policy": "disabled",
            "dataset_sha256": hashlib.sha256(Path("evals/datasets/routing_v2.jsonl").read_bytes()).hexdigest(),
            "corpus_snapshot_id": hashlib.sha256(str(DOCUMENTS).encode()).hexdigest()}
        yield runtime
