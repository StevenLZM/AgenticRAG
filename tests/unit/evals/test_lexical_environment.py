from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def files(tmp_path, monkeypatch):
    from evals import lexical_environment as mod
    home = tmp_path / "es"
    plugin, config = home / "plugins/analysis-ik", home / "config/analysis-ik"
    plugin.mkdir(parents=True)
    config.mkdir(parents=True)
    (plugin / "plugin.jar").write_bytes(b"known implementation")
    (plugin / "plugin-descriptor.properties").write_text("version=8.19.0\n")
    for name in ("main.dic", "surname.dic", "quantifier.dic", "suffix.dic", "preposition.dic", "stopword.dic"):
        (config / name).write_text("中文")
    (config / "IKAnalyzer.cfg.xml").write_text('<properties><entry key="ext_dict"></entry></properties>')
    monkeypatch.setattr(mod, "KNOWN_JARS", {"plugin.jar": hashlib.sha256(b"known implementation").hexdigest()})
    return {"provider": "native_local", "node_id": "n", "node_url": "http://localhost:9200",
            "es_home": str(home), "config_dir": str(home / "config")}, config


def test_same_versions_different_dictionary_changes_identity(files):
    from evals.lexical_environment import inspect_ik_files
    node, config = files
    before = inspect_ik_files(node, started_at_ms=9_000_000_000_000)
    (config / "main.dic").write_text("改变")
    after = inspect_ik_files(node, started_at_ms=9_000_000_000_000)
    assert before["files_sha256"] != after["files_sha256"]


def test_default_remote_switch_without_sources_is_not_remote_usage(files):
    from evals.lexical_environment import inspect_ik_files
    node, config = files
    (config / "IKAnalyzer.cfg.xml").write_text('<properties><!-- <entry key="remote_ext_dict">http://invalid</entry> --></properties>')
    proof = inspect_ik_files(node, started_at_ms=9_000_000_000_000)
    assert proof["remote_allowed"] is True
    assert proof["remote_sources"] == []
    assert proof["missing_evidence"] == []


def test_remote_dictionary_without_loaded_state_is_unverified(files):
    from evals.lexical_environment import inspect_ik_files
    node, config = files
    (config / "IKAnalyzer.cfg.xml").write_text('<properties><entry key="remote_ext_dict">http://invalid/dict</entry></properties>')
    proof = inspect_ik_files(node, started_at_ms=9_000_000_000_000)
    assert "remote_loaded_dictionary_unverifiable" in proof["missing_evidence"]


def test_config_override_and_dictionary_modified_after_start_rejected(files):
    from evals.lexical_environment import inspect_ik_files
    node, config = files
    fallback = config.parents[1] / "plugins/analysis-ik/config"
    fallback.mkdir()
    (fallback / "IKAnalyzer.cfg.xml").write_text('<properties><entry key="remote_ext_dict">http://invalid</entry></properties>')
    result = inspect_ik_files(node, started_at_ms=1)
    assert result["effective_config"] == "es_config"
    assert result["remote_sources"] == []
    assert "file_changed_after_process_start" in result["missing_evidence"]


def test_missing_extensions_and_symlink_escape_are_unverified(files, tmp_path):
    from evals.lexical_environment import inspect_ik_files
    node, config = files
    external = tmp_path / "outside.dic"
    external.write_text("秘密")
    (config / "escape.dic").symlink_to(external)
    (config / "IKAnalyzer.cfg.xml").write_text('<properties><entry key="ext_dict">escape.dic;missing.dic</entry></properties>')
    assert inspect_ik_files(node, started_at_ms=9_000_000_000_000)["missing_evidence"]


class EnvES:
    def __init__(self):
        self.indices = self.nodes = self

    async def info(self, **kwargs):
        if kwargs:
            return SimpleNamespace(body={"nodes": {"n": {"version": "8.19.0", "plugins": [{"name": "analysis-ik", "version": "8.19.0"}]}}})
        return SimpleNamespace(body={"cluster_uuid": "cluster", "version": {"number": "8.19.0", "lucene_version": "9.12"}})

    async def get_alias(self, **kwargs):
        return SimpleNamespace(body={"index": {"aliases": {"alias": {}}}})

    async def get_settings(self, **kwargs):
        return SimpleNamespace(body={"index": {"settings": {"index": {"uuid": "uuid"}}, "defaults": {}}})

    async def get_mapping(self, **kwargs):
        return SimpleNamespace(body={"index": {"mappings": {"properties": {"contextualized_content": {"type": "text",
            "fields": {"zh": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"}}}}}}})

    async def analyze(self, **kwargs):
        return SimpleNamespace(body={"tokens": [{"token": "test", "position": 0, "start_offset": 0, "end_offset": 4}]})


async def test_probe_equality_cannot_replace_artifact_hash():
    from evals.lexical_environment import capture_lexical_environment
    value = await capture_lexical_environment(es=EnvES(), index="index", alias="alias", nodes=[], implementation={"test": "hash"})
    assert value["verified"] is False
    assert "node_file_evidence_missing" in value["missing_evidence"]
    assert len(value["content"]["probes"]) == 15


async def test_missing_node_or_process_binding_is_unverified(files):
    from evals.lexical_environment import collect_local_node_evidence
    node, _ = files
    proof = await collect_local_node_evidence(node, es=EnvES())
    assert proof["verified"] is False


async def test_capture_time_not_content_drift():
    from evals.lexical_environment import capture_lexical_environment, assert_lexical_environment_unchanged
    before = await capture_lexical_environment(es=EnvES(), index="index", alias="alias", nodes=[], implementation={"test": "hash"})
    after = deepcopy(before)
    after["observed_at"] = "tomorrow"
    assert_lexical_environment_unchanged(before, after)
    after["content"]["settings"] = {}
    with pytest.raises(ValueError, match="drift|hash"):
        assert_lexical_environment_unchanged(before, after)


async def test_node_settings_do_not_persist_credentials():
    import json
    from evals.lexical_environment import capture_lexical_environment
    class SensitiveES(EnvES):
        async def info(self, **kwargs):
            result = await super().info(**kwargs)
            if kwargs:
                result.body["nodes"]["n"]["settings"] = {"xpack": {"security": {"secret": "credential-marker"}}}
            return result
    result = await capture_lexical_environment(es=SensitiveES(), index="index", alias="alias", nodes=[], implementation={})
    assert "credential-marker" not in json.dumps(result)


@pytest.mark.parametrize("name", ["main.dic", "surname.dic", "quantifier.dic", "suffix.dic", "preposition.dic", "stopword.dic"])
def test_missing_builtin_is_unverified_even_with_matching_existing_files(files, name):
    from evals.lexical_environment import inspect_ik_files
    node, config = files
    (config / name).unlink()
    result = inspect_ik_files(node, started_at_ms=9_000_000_000_000)
    assert result["missing_evidence"]


def test_deleted_extension_member_after_start_is_unverified(files):
    import time
    from evals.lexical_environment import inspect_ik_files
    node, config = files
    extension = config / "extended"
    extension.mkdir()
    member = extension / "old.dic"
    member.write_text("loaded into memory")
    (config / "IKAnalyzer.cfg.xml").write_text('<properties><entry key="ext_dict">extended</entry></properties>')
    started = time.time_ns() / 1_000_000
    member.unlink()
    result = inspect_ik_files(node, started_at_ms=started)
    assert "directory_changed_after_process_start" in result["missing_evidence"]
