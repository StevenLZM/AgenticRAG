"""Identity of effective query provider/configuration; no secrets are exported."""
import hashlib
import json
from pathlib import Path


def evaluation_config_fingerprint(settings):
    names = ("deepseek_base_url", "deepseek_protocol", "qwen_embedding_base_url", "main_model", "light_model",
             "embedding_model", "embedding_dimensions", "reranker_model", "max_concurrent_query_runs",
             "max_concurrent_llm_calls", "max_concurrent_reranks", "max_parallel_subagents_per_run",
             "max_research_rounds", "max_answer_revisions", "query_run_timeout_seconds", "max_evidence_tokens",
             "research_context_soft_limit_tokens")
    payload = {name: getattr(settings, name, None) for name in names}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def query_implementation_fingerprint():
    root = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
