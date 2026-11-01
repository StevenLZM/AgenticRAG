"""CLI planning never opens services and execution requires trusted boundaries."""

import json
import pytest
from scripts.rebuild_context_index import main
from agentic_rag.config import Settings
from agentic_rag.runtime.query_composition import build_query_snapshot


def arguments(tmp_path, command="enqueue"):
    common = dict(
        mysql_dsn="sqlite+aiosqlite:///:memory:",
        deepseek_base_url="http://localhost/v1",
        qwen_embedding_base_url="http://localhost/v1",
        context_release_enabled=True,
        context_instance_id="rebuild-1",
    )
    current = Settings(**common, index_generation="old-v1")
    target = Settings(**common, index_generation="new-v1")
    for name, config in (("current", current), ("target", target)):
        (tmp_path / f"{name}.json").write_text(config.model_dump_json())
    manifest = dict(
        source_generation="old-v1",
        target_generation="new-v1",
        target_snapshot_id=build_query_snapshot(target).snapshot_id,
        documents=[],
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return [
        command,
        "--source",
        "old-v1",
        "--target",
        "new-v1",
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--config",
        str(tmp_path / "current.json"),
        "--target-config",
        str(tmp_path / "target.json"),
    ]


def test_default_cli_only_reports_unverified_plan(tmp_path, monkeypatch, capsys):
    import agentic_rag.bootstrap

    monkeypatch.setattr(
        agentic_rag.bootstrap,
        "build_container",
        lambda *a: pytest.fail("plan touched live dependencies"),
    )
    assert main(arguments(tmp_path)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "plan" and result["readiness"] == "not_checked_offline"


def test_execute_release_without_authority_fails_before_opening_services(
    tmp_path, monkeypatch
):
    import agentic_rag.bootstrap

    monkeypatch.setattr(
        agentic_rag.bootstrap,
        "build_container",
        lambda *a: pytest.fail("unauthorized execute opened dependencies"),
    )
    with pytest.raises(ValueError, match="authority"):
        main(arguments(tmp_path, "release") + ["--journal", "release-1", "--execute"])


def test_recover_execution_requires_journal_and_external_proof_before_clients(
    tmp_path, monkeypatch, capsys
):
    import agentic_rag.bootstrap

    monkeypatch.setattr(
        agentic_rag.bootstrap,
        "build_container",
        lambda *a: pytest.fail("recovery opened services without proof"),
    )
    args = arguments(tmp_path, "recover-execution")
    with pytest.raises(ValueError, match="journal"):
        main(args)
    assert main(args + ["--journal", "journal-1"]) == 0
    assert "settled ES" in capsys.readouterr().out
    with pytest.raises(ValueError, match="authority"):
        main(args + ["--journal", "journal-1", "--execute"])
