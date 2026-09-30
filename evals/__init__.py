"""Offline, deterministic evaluation contracts for Agentic RAG.

The package is deliberately independent from the live QueryGraph and model
providers.  It contains only fixed dataset schemas and pure metric helpers.
"""

from evals.metrics import (
    aggregate_loop_metrics,
    aggregate_security_metrics,
    compute_loop_metrics,
    compute_security_metrics,
    loop_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
    security_metrics,
)
from evals.models import (
    EvaluationCase,
    IngestionFidelityCase,
    SecurityCase,
    load_jsonl_dataset,
    validate_dataset_directory,
    validate_dataset_rows,
)
from evals.ragas_adapter import (
    RagasAdapter,
    RagasEvaluation,
    RagasUnavailable,
    normalize_ragas_result,
)
from evals.report import MixedSnapshotError, build_summary, write_summary

__all__ = [
    "EvaluationCase",
    "EvalCaseResult",
    "EvalRunner",
    "IngestionFidelityCase",
    "MixedSnapshotError",
    "SecurityCase",
    "RagasAdapter",
    "RagasEvaluation",
    "RagasUnavailable",
    "SnapshotMismatchError",
    "aggregate_loop_metrics",
    "aggregate_security_metrics",
    "build_summary",
    "compute_loop_metrics",
    "compute_security_metrics",
    "load_jsonl_dataset",
    "loop_metrics",
    "mrr",
    "ndcg_at_k",
    "normalize_ragas_result",
    "recall_at_k",
    "security_metrics",
    "validate_dataset_directory",
    "validate_dataset_rows",
    "write_summary",
]


def __getattr__(name: str) -> object:
    """Lazily expose runner classes without importing ``evals.run`` for the CLI."""

    if name in {"EvalCaseResult", "EvalRunner", "SnapshotMismatchError"}:
        from evals import run

        return getattr(run, name)
    raise AttributeError(name)
