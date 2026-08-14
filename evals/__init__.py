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

__all__ = [
    "EvaluationCase",
    "IngestionFidelityCase",
    "SecurityCase",
    "aggregate_loop_metrics",
    "aggregate_security_metrics",
    "compute_loop_metrics",
    "compute_security_metrics",
    "load_jsonl_dataset",
    "loop_metrics",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "security_metrics",
    "validate_dataset_directory",
    "validate_dataset_rows",
]
