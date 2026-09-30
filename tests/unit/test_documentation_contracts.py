"""Stable acceptance contracts for the Chinese operational documentation."""

from pathlib import Path


OPERATIONS = Path("docs/local-operations.md")
PROGRESS = Path("docs/development-progress.md")
ISSUES = Path("docs/issue-log.md")
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
        "ingestion-v2",
        "index-v3",
        "rebuild_preflight.py",
        "source-span",
        "dry-run",
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
        "ingestion-v2",
        "index-v3",
        "rebuild_preflight.py",
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


def test_docs_keep_process_api_acceptance_and_checkpoint_boundaries_accurate() -> None:
    """Catch stale console, verifier, or research-state operating guidance."""
    operations = OPERATIONS.read_text(encoding="utf-8")
    progress = PROGRESS.read_text(encoding="utf-8")
    roadmap = ROADMAP.read_text(encoding="utf-8")

    for document in (operations, progress):
        assert "三个独立终端" in document
        assert "阻塞进程" in document

    for term in (
        "`POST /v1/query`",
        "同步 wrapper",
        "`POST /v1/query-runs`",
        "异步 API",
    ):
        assert term in operations

    for document in (operations, progress, roadmap):
        assert "QueryState/SQLite checkpoint" in document
        assert "QueryState/Run" not in document

    for document in (operations, progress, roadmap):
        assert "run_real_query_acceptance.py" in document
        assert "console_acceptance_passed" in document
        assert "在 EvalRunner 后执行" in document
        assert "EvalRunner console gate" not in document
        assert "通用验证器" in document
        assert "不单独校验" in document


def test_progress_and_issue_log_keep_a_reproducible_handoff_snapshot() -> None:
    """Keep the next development session anchored to evidence and open gates."""
    progress = PROGRESS.read_text(encoding="utf-8")
    issues = ISSUES.read_text(encoding="utf-8")

    assert "[问题与修改日志](./issue-log.md)" in progress
    for term in (
        "2026-08-28",
        "679f4f2",
        "工作树存在未提交改动",
        "642 passed",
        "SubagentDispatcher",
        "research_attempt_count",
        "Evaluation PASS",
        "生产鉴权/RBAC",
        "reranker 分数标定",
    ):
        assert term in progress

    for term in (
        "01a045b2-2bc0-73a9-abbc-c8dd9edd06af",
        "provider_outage",
        "agent_events",
        "Query Outbox",
        "Query Worker",
        "Elasticsearch 活动 Alias",
        "MySQL UTC",
        "SubagentDispatcher",
        "Todo 初始创建",
        "research_attempt_count",
        "LLM 诊断字段",
        "生产鉴权/RBAC",
        "reranker 分数标定",
    ):
        assert term in issues
