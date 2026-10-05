import argparse
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

import pytest

from evals.gold_v2_models import canonical_hash
from tests.unit.evals.test_gold_rebinding import inputs  # noqa: F401


def test_cli_has_no_legacy_or_paid_fallback():
    from evals.ik_evaluation_command import main
    with pytest.raises(SystemExit) as result:
        main(["bm25"])
    assert result.value.code == 2
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0


def test_no_corpus_writes_or_query_posts():
    from evals.ik_evaluation_command import ReadOnlyES, PhaseBudget
    class Forbidden:
        def index(self, **kwargs):
            pytest.fail("write reached ES")
    wrapped = ReadOnlyES(Forbidden(), PhaseBudget("audit", 250, 300))
    with pytest.raises(ValueError):
        wrapped.index
    with pytest.raises(ValueError):
        wrapped.perform_request


@pytest.fixture
def command_fixture(inputs, tmp_path, monkeypatch):  # noqa: F811
    from evals import ik_evaluation_command as mod
    from evals.gold_rebinding import publish_rebound_bundle
    publish_rebound_bundle(**inputs)
    controls = {"fail_pre": False, "fail_post": False, "preflights": 0, "searches": 0}

    class ES:
        def options(self, **kwargs):
            assert kwargs.get("max_retries", 0) == 0
            return self

        async def search(self, **kwargs):
            controls["searches"] += 1
            return SimpleNamespace(body={"timed_out": False, "_shards": {"failed": 0}, "took": 1,
                "hits": {"hits": [{"_id": "c", "_score": 1.0, "_source": {"id": "c", "parent_id": "p", "user_id": "u",
                    "document_id": "d", "document_version_id": "v", "content": "事实", "ast_locator": "{}"}}]}})

    @asynccontextmanager
    async def resources(args):
        yield SimpleNamespace(model_dump=lambda **kwargs: {}), None, ES(), None

    async def preflight(*args, **kwargs):
        controls["preflights"] += 1
        if controls["fail_pre"] or (controls["fail_post"] and controls["preflights"] % 2 == 0):
            raise mod.PreflightError("fixture_unverified")
        content = {"test": "environment"}
        return inputs["evidence"], {"verified": True, "content": content, "environment_sha256": canonical_hash(content)}

    monkeypatch.setattr(mod, "resources", resources)
    monkeypatch.setattr(mod, "_preflight", preflight)
    monkeypatch.setattr(mod, "select_paired_cases", lambda cases: cases)
    args = argparse.Namespace(command="bm25", bundle=inputs["output"], output_dir=tmp_path / "experiment",
        es_url="http://localhost:9200", user_id="u", settings_file=tmp_path / "settings", artifact_root=tmp_path,
        node_evidence_config=tmp_path / "nodes.json")
    args.node_evidence_config.write_text("[]")
    return args, controls


async def test_preflight_failure_sends_no_bm25(command_fixture):
    from evals.ik_evaluation_command import run
    args, controls = command_fixture
    controls["fail_pre"] = True
    assert await run(args) == 1
    assert controls["searches"] == 0
    assert json.loads((args.output_dir / "preflight-failure.json").read_text())["verified"] is False


async def test_postflight_drift_invalidates_pairs(command_fixture):
    from evals.ik_evaluation_command import run
    args, controls = command_fixture
    controls["fail_post"] = True
    assert await run(args) == 1
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["valid_pairs"] == 0 and controls["searches"] == 2


async def test_resume_rechecks_even_when_finished_and_private_outputs(command_fixture):
    from evals.ik_evaluation_command import run
    args, controls = command_fixture
    assert await run(args) == 0
    ledger = (args.output_dir / "ledger.json").read_bytes()
    args.command, args.experiment = "resume", args.output_dir
    assert await run(args) == 0
    assert controls["searches"] == 2 and controls["preflights"] == 4
    assert (args.experiment / "ledger.json").read_bytes() == ledger
    assert (args.experiment / "report.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((args.experiment / "audit-resume.json").read_text())["new_bm25_searches"] == 0


def test_phase_request_budget_is_bounded():
    from evals.ik_evaluation_command import PhaseBudget
    budget = PhaseBudget("audit", 2, 300)
    budget.consume("info")
    budget.consume("info")
    with pytest.raises(ValueError, match="budget"):
        budget.consume("info")


@pytest.mark.parametrize("initial_postflight", ["failed", "missing"])
async def test_resume_never_rehabilitates_unverified_past_requests(command_fixture, monkeypatch, initial_postflight):
    from evals import ik_evaluation_command as mod
    args, controls = command_fixture
    preflight = mod._preflight
    if initial_postflight == "failed":
        controls["fail_post"] = True
        assert await mod.run(args) == 1
        controls["fail_post"] = False
    else:
        class Crash(BaseException):
            pass

        async def crash_before_postflight(*a, **kw):
            if controls["preflights"] == 1:
                raise Crash()
            return await preflight(*a, **kw)

        monkeypatch.setattr(mod, "_preflight", crash_before_postflight)
        with pytest.raises(Crash):
            await mod.run(args)
        monkeypatch.setattr(mod, "_preflight", preflight)
    previous_ledger = (args.output_dir / "ledger.json").read_bytes()
    args.command, args.experiment = "resume", args.output_dir
    assert await mod.run(args) == 1
    report = json.loads((args.experiment / "report.json").read_text())
    assert report["valid_pairs"] == 0 and report["completion_verified"] is False
    assert (args.output_dir / "ledger.json").read_bytes() == previous_ledger
    assert controls["searches"] == 2


async def test_analyzer_comparison_persists_and_resume_uses_saved_choice(command_fixture):
    from evals.ik_evaluation_command import run
    args, controls = command_fixture
    args.comparison = "ik-smart-vs-max"
    assert await run(args) == 0
    identity = json.loads((args.output_dir / "experiment.json").read_text())
    assert identity["common"]["config"]["comparison"] == "ik-smart-vs-max"
    assert set(identity["arms"]) == {"ik", "ik_max_word"}
    args.command, args.experiment = "resume", args.output_dir
    args.comparison = "standard-vs-ik"  # resume takes its mode from the saved experiment
    assert await run(args) == 0
    assert controls["searches"] == 2
    assert json.loads((args.experiment / "report.json").read_text())["arm_order"] == ["ik", "ik_max_word"]


def test_cli_accepts_explicit_analyzer_comparison(monkeypatch, tmp_path):
    from evals import ik_evaluation_command as mod
    captured = []

    async def capture(args):
        captured.append(args)
        return 0

    monkeypatch.setattr(mod, "run", capture)
    options = ["bm25", "--comparison", "ik-smart-vs-max", "--es-url", "http://localhost:9200", "--user-id", "u"]
    for name in ("settings-file", "artifact-root", "bundle", "node-evidence-config", "output-dir"):
        options.extend(["--" + name, str(tmp_path / name)])
    assert mod.main(options) == 0
    assert captured[0].comparison == "ik-smart-vs-max"
