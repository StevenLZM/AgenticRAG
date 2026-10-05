"""Explicit, read-only IK evaluation commands. No API/model/legacy fallback."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import time
from urllib.parse import urlsplit

from evals.gold_rebinding import _json, _write, file_sha256, load_rebound_bundle, publish_rebound_bundle
from evals.gold_v2_models import CorpusSnapshot
from evals.lexical_environment import capture_lexical_environment, assert_lexical_environment_unchanged
from evals.migration_evidence import capture_corpus_evidence, capture_migration_evidence, assert_corpus_unchanged
from evals.paired_bm25 import run_paired_bm25
from evals.paired_retrieval_identity import COMPARISONS, build_paired_identity, select_paired_cases, validate_identity
from evals.paired_retrieval_ledger import PairedLedger
from evals.paired_retrieval_report import build_paired_report, render_paired_report
from evals.report import atomic_write_text


class PreflightError(ValueError):
    pass


class PhaseBudget:
    def __init__(self, label, limit, wall_seconds):
        self.label, self.limit, self.wall_seconds = label, limit, wall_seconds
        self.started, self.calls = time.monotonic(), Counter()

    def consume(self, method):
        if sum(self.calls.values()) >= self.limit or self.remaining <= 0:
            raise ValueError(self.label + " budget exhausted")
        self.calls[method] += 1

    @property
    def remaining(self):
        return max(0, self.wall_seconds - (time.monotonic() - self.started))

    def summary(self):
        return {"phase": self.label, "calls": dict(self.calls), "total": sum(self.calls.values()),
                "elapsed_seconds": time.monotonic() - self.started, "limit": self.limit}


class ReadOnlyES:
    """Allow only explicit reads and cleanup of read-created scroll contexts."""
    METHODS = {"": {"info", "search", "scroll", "clear_scroll"},
               "indices": {"get_alias", "get_settings", "get_mapping", "analyze"}, "nodes": {"info"}}

    def __init__(self, client, budget, namespace=""):
        self._client, self._budget, self._namespace = client, budget, namespace

    def options(self, **kwargs):
        kwargs.update(max_retries=0, retry_on_timeout=False, request_timeout=30)
        return ReadOnlyES(self._client.options(**kwargs), self._budget, self._namespace)

    def __getattr__(self, name):
        if not self._namespace and name in {"indices", "nodes"}:
            return ReadOnlyES(getattr(self._client, name), self._budget, name)
        if name not in self.METHODS[self._namespace]:
            raise ValueError("ES operation outside read-only protocol")

        async def call(**kwargs):
            self._budget.consume(".".join(filter(None, (self._namespace, name))))
            async with asyncio.timeout(min(30, self._budget.remaining)):
                return await getattr(self._client, name)(**kwargs)
        return call


def _endpoint(value):
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("explicit credential-free ES URL required")
    return (parsed.scheme, "127.0.0.1" if parsed.hostname == "localhost" else parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80), parsed.path.rstrip("/"))


@asynccontextmanager
async def resources(args):
    from agentic_rag.config import Settings
    from agentic_rag.persistence.artifacts import LocalArtifactStore
    from elasticsearch import AsyncElasticsearch
    from sqlalchemy.ext.asyncio import create_async_engine
    if not args.settings_file.is_file() or not args.artifact_root.is_dir():
        raise ValueError("explicit settings file and existing artifact root required")
    _endpoint(args.es_url)
    settings = Settings(_env_file=args.settings_file, artifact_root=args.artifact_root, elasticsearch_url=args.es_url)
    if settings.default_user_id != args.user_id or settings.index_generation != "index-v4":
        raise ValueError("settings user/generation mismatch")
    engine = create_async_engine(settings.mysql_dsn)
    try:
        async with AsyncElasticsearch(args.es_url, request_timeout=30, max_retries=0, retry_on_timeout=False) as es:
            yield settings, engine, es, LocalArtifactStore(args.artifact_root)
    finally:
        await engine.dispose()


def implementation_fingerprint():
    root = Path(__file__).resolve().parent.parent
    paths = sorted([*root.joinpath("evals").rglob("*.py"), *root.joinpath("src").rglob("*.py"),
                    root / "scripts/build_formal_eval_gold.py", root / "scripts/migrate_ik_index.py"])
    versions = {}
    for name in ("elasticsearch", "elastic-transport", "SQLAlchemy", "asyncmy", "pydantic", "pypdf", "openpyxl", "python-docx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                      timeout=10, check=True).stdout.strip(),
            "files": {str(p.relative_to(root)): file_sha256(p) for p in paths},
            "python": platform.python_version(), "dependencies": versions}


def _settings_identity(settings):
    # Deliberate allowlist: never serialize credentials, DSNs, arbitrary headers or MCP configs.
    values = settings.model_dump(mode="json")
    keys = ("default_user_id", "index_generation", "lexical_analysis", "main_model", "light_model", "embedding_model",
            "embedding_dimensions", "reranker_model", "parser_version", "ingestion_pipeline_version", "max_research_rounds",
            "max_evidence_tokens", "query_run_timeout_seconds", "max_concurrent_query_runs", "max_concurrent_llm_calls")
    return {key: values[key] for key in keys if key in values}


def _scope(snapshot, args):
    if snapshot.scope["user_id"] != args.user_id or _endpoint(snapshot.scope["es_url"]) != _endpoint(args.es_url):
        raise ValueError("snapshot user/ES scope mismatch")
    indices = snapshot.scope["concrete_indices"]
    generations = {d["index_generation"] for d in snapshot.documents}
    if indices != ["agenticrag-children-index-v4"] or generations != {"index-v4"}:
        raise ValueError("first batch requires physical v4 corpus")
    return indices[0], "index-v4"


async def _preflight(snapshot, baseline, args, *, engine, es, artifacts, implementation, budget):
    index, generation = _scope(snapshot, args)
    try:
        async with asyncio.timeout(budget.remaining):
            _, corpus = await capture_corpus_evidence(snapshot, engine=engine, es=ReadOnlyES(es, budget), artifacts=artifacts,
                                                       index=index, generation=generation)
            assert_corpus_unchanged(baseline, corpus)
            environment = await capture_lexical_environment(es=ReadOnlyES(es, budget), index=index,
                alias=snapshot.scope["alias"], nodes=_json(args.node_evidence_config), implementation=implementation)
            if environment["verified"] is not True:
                raise PreflightError("lexical_evidence_unverified")
            return corpus, environment
    except (ValueError, KeyError, OSError, TimeoutError) as error:
        if isinstance(error, PreflightError):
            raise
        raise PreflightError(type(error).__name__) from error


def _failure(root, error, *, resume):
    if not resume:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    name = "resume-preflight-failure.json" if resume else "preflight-failure.json"
    _write(root / name, {"verified": False, "error_type": type(error).__name__,
                         "reason": str(error) if isinstance(error, PreflightError) else "input_or_runtime_failure"})


async def _paired(args, settings, engine, es, artifacts):
    resume = args.command == "resume"
    root = args.experiment if resume else args.output_dir
    if resume:
        saved = _json(root / "experiment.json")
        validate_identity(saved)
        bundle = Path(saved["common"]["config"]["bundle_path"])
    else:
        if root.exists() or root.is_symlink():
            raise ValueError("new experiment output already exists")
        bundle = args.bundle
    snapshot, all_cases, _, audit = load_rebound_bundle(bundle)
    cases = select_paired_cases(all_cases)
    baseline = _json(bundle / "migration-evidence.json")
    implementation = implementation_fingerprint()
    pre_budget = PhaseBudget("preflight", 250, 300)
    try:
        before, lexical_before = await _preflight(snapshot, baseline, args, engine=engine, es=es, artifacts=artifacts,
                                                   implementation=implementation, budget=pre_budget)
    except Exception as error:
        _failure(root, error, resume=resume)
        return 1
    config = {"top_k": 40, "request_limit": 200, "timeout_s": 30, "wall_s": 900, "user_id": args.user_id,
              "index": snapshot.scope["concrete_indices"][0], "index_generation": snapshot.documents[0]["index_generation"],
              "es_endpoint": _endpoint(args.es_url), "bundle_path": str(bundle.resolve()),
              "settings": _settings_identity(settings), "artifact_root": str(args.artifact_root.resolve()),
              "node_config_sha256": file_sha256(args.node_evidence_config)}
    # Lists survive JSON round-trips; experiment identity must too.
    config["es_endpoint"] = list(config["es_endpoint"])
    if resume:
        if "comparison" in saved["common"]["config"]:
            config["comparison"] = saved["common"]["config"]["comparison"]
    elif getattr(args, "comparison", "standard-vs-ik") != "standard-vs-ik":
        config["comparison"] = args.comparison
    identity = build_paired_identity(bundle_audit=audit, environment=lexical_before, cases=cases, config=config,
                                     implementation=implementation)
    with PairedLedger.open(root, identity, resume=resume) as ledger:
        initial = len(ledger.records())
        pre_name = "lexical-pre-resume.json" if resume else "lexical-pre.json"
        _write(root / pre_name, lexical_before)
        retrieval_budget = PhaseBudget("retrieval", 200 - initial, 900)
        records = await run_paired_bm25(es=ReadOnlyES(es, retrieval_budget), snapshot=snapshot, cases=cases,
                                       identity=identity, ledger=ledger)
        post_budget = PhaseBudget("postflight", 250, 300)
        verified = {"pre_corpus": True, "pre_lexical": True, "post_corpus": False, "post_lexical": False}
        try:
            after, lexical_after = await _preflight(snapshot, baseline, args, engine=engine, es=es, artifacts=artifacts,
                implementation=implementation, budget=post_budget)
            assert_corpus_unchanged(before, after)
            assert_lexical_environment_unchanged(lexical_before, lexical_after)
            if implementation != implementation_fingerprint():
                raise PreflightError("implementation_changed_during_run")
            _write(root / ("lexical-post-resume.json" if resume else "lexical-post.json"), lexical_after)
            verified.update(post_corpus=True, post_lexical=True)
        except Exception as error:
            verified["post_error_type"] = type(error).__name__
        ledger.attest_current_interval(verified)
        records = ledger.records()
        report = build_paired_report(cases=cases, records=records, identity=identity, verification=verified)
        _write(root / "report.json", report)
        atomic_write_text(root / "report.md", render_paired_report(report))
        (root / "report.md").chmod(0o600)
        _write(root / ("audit-resume.json" if resume else "audit-first.json"), {
            "new_bm25_searches": sum(retrieval_budget.calls.values()), "new_attempts": len(records) - initial,
            "query_posts": 0, "model_calls": 0, "verification": verified,
            "phases": [b.summary() for b in (pre_budget, retrieval_budget, post_budget)],
            "ledger_sha256": file_sha256(root / "ledger.json"), "results_sha256": file_sha256(root / "results.json")})
        return 0 if report["completion_verified"] else 1


async def run(args: argparse.Namespace) -> int:
    try:
        async with resources(args) as (settings, engine, es, artifacts):
            if args.command != "rebind":
                return await _paired(args, settings, engine, es, artifacts)
            if args.output_dir.exists() or args.output_dir.is_symlink():
                raise ValueError("rebind output exists")
            old = CorpusSnapshot.model_validate(_json(args.source_snapshot))
            if old.scope["user_id"] != args.user_id or _endpoint(old.scope["es_url"]) != _endpoint(args.es_url):
                raise ValueError("source snapshot scope mismatch")
            budget = PhaseBudget("rebind", 250, 300)
            async with asyncio.timeout(300):
                target, evidence = await capture_migration_evidence(old, _json(args.journal), engine=engine,
                    es=ReadOnlyES(es, budget), artifacts=artifacts)
                publish_rebound_bundle(source_snapshot=args.source_snapshot, source_gold=args.source_gold,
                    source_specs=args.source_specs, journal_path=args.journal, target=target, evidence=evidence, output=args.output_dir)
            _write(args.output_dir / "capture-budget.json", budget.summary())
            return 0
    except (ValueError, KeyError, OSError) as error:
        print("IK evaluation refused: " + type(error).__name__)
        return 2
    except Exception as error:
        print("IK evaluation incomplete: " + type(error).__name__)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only v4 gold rebind / real BM25 paired diagnostics; no LLM calls")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("rebind", "bm25", "resume"):
        command = sub.add_parser(name)
        if name == "bm25":
            command.add_argument("--comparison", choices=tuple(COMPARISONS), default="standard-vs-ik",
                help="ik-smart-vs-max changes only the zh query analyzer; production settings stay unchanged")
        for option in ("settings-file", "artifact-root", "es-url", "user-id"):
            command.add_argument("--" + option, type=Path if option in {"settings-file", "artifact-root"} else str, required=True)
        options = {"rebind": ("source-snapshot", "source-gold", "source-specs", "journal", "output-dir"),
                   "bm25": ("bundle", "node-evidence-config", "output-dir"),
                   "resume": ("experiment", "node-evidence-config")}[name]
        for option in options:
            command.add_argument("--" + option, type=Path, required=True)
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
