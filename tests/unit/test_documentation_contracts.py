"""Stable acceptance contracts for the Chinese operational documentation."""

from pathlib import Path


OPERATIONS = Path("docs/local-operations.md")
PROGRESS = Path("docs/development-progress.md")
ROADMAP = Path(
    "docs/superpowers/plans/2026-08-04-agentic-rag-implementation-roadmap.md"
)


def test_chinese_operations_docs_cover_query_runtime_and_console_contracts() -> None:
    """Catch removal of the production runtime and console operating contract."""
    operations = OPERATIONS.read_text(encoding="utf-8")

    required_terms = (
        "Agentic RAG 控制台",
        "Query Outbox",
        "Query Worker",
        "MySQL 同事务",
        "Redis 领取",
        "租约",
        "心跳",
        "恢复",
        "重试/DLQ",
        "终态 ACK",
        "SubagentDispatcher",
        "Todo 初始创建",
        "Todo 追加",
        "research_attempt_count",
        "Mem0 默认启用",
        "memory_provider_degraded",
        "MODEL_REPAIR_EXHAUSTED",
        "CIRCUIT_OPEN",
        "WORKER_DLQ",
        "真实 Graph/API",
        "当前 snapshot",
        "client provenance",
        "备份",
    )
    required_commands_and_paths = (
        "python scripts/run_api.py",
        "python scripts/run_query_worker.py",
        "curl http://127.0.0.1:8000/",
        "scripts/verify_acceptance.py",
        "POST /v1/query-runs",
    )

    for term in (*required_terms, *required_commands_and_paths):
        assert term in operations


def test_progress_and_roadmap_record_mainline_completion_and_release_work() -> None:
    """Catch stale worktree claims or loss of the remaining production gates."""
    progress = PROGRESS.read_text(encoding="utf-8")
    roadmap = ROADMAP.read_text(encoding="utf-8")

    for term in (
        "当前 `main`",
        "SubagentDispatcher",
        "Todo 初始创建",
        "Todo 追加",
        "research_attempt_count",
        "真实 Graph/API",
        "当前 snapshot",
        "client provenance",
        "Mem0",
        "恢复",
        "备份",
        "生产鉴权/RBAC",
        "reranker 分数标定",
    ):
        assert term in progress

    for term in (
        "已合并到 `main`",
        "真实 Graph/API",
        "当前 snapshot",
        "client provenance",
        "Mem0",
        "恢复",
        "备份",
        "生产鉴权/RBAC",
        "reranker 分数标定",
    ):
        assert term in roadmap
