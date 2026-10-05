"""Content-addressed, fail-closed IK installation evidence; never install a plugin."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shlex
import subprocess
from typing import Any
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET
import zipfile

from evals.gold_v2_models import canonical_hash

KNOWN_JARS = {
    "elasticsearch-analysis-ik-8.19.0.jar": "a8bb79edbcbc5bc778b212daf41541331febdd9eb4358b276f4a8f79a5427ab0",
    "ik-core-1.0.jar": "2d4050f58394ee513c8546463ac3ede25f2a66518a590a0fbaa3b8a5dc777e80",
}
PACKAGE_SHA256 = "bcacbcea8dab5555cc8ad4bc59c0403cd583ed1ee169b7764e5bbb81464c7673"
PROBES = ("中华人民共和国劳动合同", "星河智采量子枢纽", "Redis缓存A17", "2026年10月3日报销上限560元", "研究研究生生命起源")
BUILTIN_DICTIONARIES = ("main.dic", "surname.dic", "quantifier.dic", "suffix.dic", "preposition.dic", "stopword.dic")


def inspect_ik_files(node: dict[str, Any], *, started_at_ms: int) -> dict[str, Any]:
    home, conf = Path(node["es_home"]).resolve(), Path(node["config_dir"]).resolve()
    plugin = home / "plugins/analysis-ik"
    primary, fallback = conf / "analysis-ik", plugin / "config"
    effective = primary if (primary / "IKAnalyzer.cfg.xml").exists() else fallback
    missing, entries, directories, remote, package_hash = [], {}, {}, [], None
    roots = (plugin, conf)

    def read_directory(path):
        resolved = path.resolve(strict=True)
        if not any(resolved.is_relative_to(root) for root in roots):
            raise ValueError("directory_outside_declared_roots")
        before = path.stat()
        members = sorted(p.name for p in path.iterdir())
        after = path.stat()
        if (before.st_mtime_ns, before.st_ctime_ns) != (after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("directory_changed_during_read")
        if max(after.st_mtime, after.st_ctime) * 1000 > started_at_ms:
            missing.append("directory_changed_after_process_start")
        prefix, base = ("plugin", plugin) if path.is_relative_to(plugin) else ("config", conf)
        directories[prefix + "/" + path.relative_to(base).as_posix()] = {
            "members": members, "mtime_ns": after.st_mtime_ns, "ctime_ns": after.st_ctime_ns, "resolved_path": str(resolved)}

    def read(path):
        resolved = path.resolve(strict=True)
        if not any(resolved.is_relative_to(r) for r in roots):
            raise ValueError("file_outside_declared_roots")
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_mtime_ns, before.st_ctime_ns, before.st_size) != (after.st_mtime_ns, after.st_ctime_ns, after.st_size):
            raise ValueError("file_changed_during_read")
        if max(after.st_mtime, after.st_ctime) * 1000 > started_at_ms:
            missing.append("file_changed_after_process_start")
        prefix, base = ("plugin", plugin) if path.is_relative_to(plugin) else ("config", conf)
        key = prefix + "/" + path.relative_to(base).as_posix()
        entries[key] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                        "mtime_ns": after.st_mtime_ns, "ctime_ns": after.st_ctime_ns,
                        "resolved_path": str(resolved)}
        return data

    try:
        # Bind absence as well as presence: deleting a loaded dictionary changes
        # its parent directory even when all surviving files predate startup.
        read_directory(conf)
        for root in (plugin, primary):
            if root.is_dir():
                read_directory(root)
            for path in sorted(root.rglob("*")):
                if path.is_dir():
                    read_directory(path)
                elif path.is_file() or path.is_symlink():
                    read(path)
        config = ET.fromstring(read(effective / "IKAnalyzer.cfg.xml"))
        for name in BUILTIN_DICTIONARIES:
            read(effective / name)
        if config.tag != "properties":
            raise ValueError("unknown_dictionary_configuration")
        properties = {}
        for element in config.findall("entry"):
            key = element.attrib["key"]
            if key in properties:
                raise ValueError("duplicate_dictionary_configuration")
            properties[key] = (element.text or "").strip()
        for key in ("remote_ext_dict", "remote_ext_stopwords"):
            if properties.get(key):
                # URLs may carry credentials. Identity needs the hash, not the URL.
                remote.append({"key": key, "sha256": hashlib.sha256(properties[key].encode()).hexdigest()})
        if remote:
            missing.append("remote_loaded_dictionary_unverifiable")
        for key in ("ext_dict", "ext_stopwords"):
            for name in properties.get(key, "").split(";"):
                if name.strip():
                    path = effective / name.strip()
                    if path.is_dir():
                        read_directory(path)
                    members = sorted(path.rglob("*")) if path.is_dir() else [path]
                    for member in members:
                        if member.is_dir():
                            read_directory(member)
                        elif member.is_file() or not member.exists():
                            read(member)
        for name, digest in KNOWN_JARS.items():
            if entries.get("plugin/" + name, {}).get("sha256") != digest:
                missing.append("unknown_plugin_implementation")
        if node.get("package_path"):
            package = Path(node["package_path"])
            package_hash = hashlib.sha256(package.read_bytes()).hexdigest()
            if package_hash != PACKAGE_SHA256:
                missing.append("unknown_plugin_package")
            else:
                with zipfile.ZipFile(package) as archive:
                    expected = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()
                                if not n.endswith("/") and not n.startswith("config/")}
                installed = {k.removeprefix("plugin/"): v["sha256"] for k, v in entries.items()
                             if k.startswith("plugin/") and not k.startswith("plugin/config/")}
                if expected != installed:
                    missing.append("installed_artifacts_differ_from_package")
    except (OSError, ValueError, ET.ParseError, KeyError, zipfile.BadZipFile) as error:
        missing.append("file_evidence_unavailable:" + type(error).__name__)
    return {"files": entries, "files_sha256": canonical_hash(entries), "directories": directories,
            "directories_sha256": canonical_hash(directories), "package_sha256": package_hash,
            "effective_config": "es_config" if effective == primary else "plugin_config",
            "remote_allowed": True, "remote_sources": remote, "missing_evidence": sorted(set(missing))}


def _run(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
    if result.returncode:
        raise ValueError("process_evidence_unavailable")
    return result.stdout


def _process_binding(node, metadata):
    pid = metadata["process"]["id"]
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid_process_id")
    port = urlsplit(node["node_url"]).port or 9200
    listening = _run(["lsof", "-Pan", "-p", str(pid), "-iTCP:" + str(port), "-sTCP:LISTEN", "-Fp"])
    if "p" + str(pid) not in listening.splitlines():
        raise ValueError("listener_process_mismatch")
    # Modern ES has a launcher parent; inspect only this PID and its direct parent.
    child = _run(["ps", "-p", str(pid), "-o", "ppid=", "-o", "command="]).strip()
    parent_text, argv = child.split(None, 1)
    parent = int(parent_text)
    candidates = [argv]
    if parent > 1:
        candidates.append(_run(["ps", "-p", str(parent), "-o", "command="]).strip())
    expected = {"-Des.path.home": str(Path(node["es_home"]).resolve()),
                "-Des.path.conf": str(Path(node["config_dir"]).resolve())}
    matched = False
    for text in candidates:
        values = dict(token.split("=", 1) for token in shlex.split(text) if token.startswith("-Des.path.") and "=" in token)
        if all(k in values and str(Path(values[k]).resolve()) == v for k, v in expected.items()):
            matched = True
    if not matched:
        raise ValueError("process_home_config_not_bound")
    opened = _run(["lsof", "-p", str(pid), "-Fn"])
    jars = sorted((Path(node["es_home"]) / "plugins/analysis-ik").glob("*.jar"))
    if not jars or any("n" + str(p.resolve()) not in opened.splitlines() for p in jars):
        raise ValueError("loaded_plugin_jars_not_bound")
    return {"pid": pid, "parent_pid": parent, "method": "node_process_listener_launcher_open_jars"}


async def collect_local_node_evidence(node: dict[str, Any], *, es) -> dict[str, Any]:
    result = {"node_id": node.get("node_id"), "verified": False, "missing_evidence": []}
    try:
        if node.get("provider") != "native_local" or urlsplit(node["node_url"]).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("unsupported_node_provider")
        response = (await es.nodes.info(node_id="_local", metric=["settings", "process", "jvm", "plugins"])).body
        if set(response["nodes"]) != {node["node_id"]}:
            raise ValueError("node_identity_mismatch")
        metadata = response["nodes"][node["node_id"]]
        plugin = [p for p in metadata["plugins"] if p["name"] == "analysis-ik"]
        if len(plugin) != 1 or plugin[0]["version"] != "8.19.0" or metadata["version"] != "8.19.0":
            raise ValueError("unsupported_plugin_version")
        binding = _process_binding(node, metadata)
        files = inspect_ik_files(node, started_at_ms=metadata["jvm"]["start_time_in_millis"])
        result.update(files, process=binding, started_at_ms=metadata["jvm"]["start_time_in_millis"])
        result["verified"] = not result["missing_evidence"]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result["missing_evidence"].append("node_process_binding_unavailable:" + type(error).__name__)
    return result


async def capture_lexical_environment(*, es, index: str, alias: str, nodes: list[dict[str, Any]],
                                       implementation: dict[str, str]) -> dict[str, Any]:
    cluster = (await es.info()).body
    all_nodes = (await es.nodes.info(metric=["plugins", "settings"])).body
    mapping = (await es.indices.get_mapping(index=index)).body
    settings = (await es.indices.get_settings(index=index, include_defaults=True)).body
    aliases = (await es.indices.get_alias(name=alias)).body
    missing = []
    proofs = [await collect_local_node_evidence(node, es=es) for node in nodes]
    if len(nodes) != len({n["node_id"] for n in nodes}) or {n["node_id"] for n in nodes} != set(all_nodes["nodes"]):
        missing.append("node_file_evidence_missing")
    for proof in proofs:
        missing.extend(proof["missing_evidence"])
    if set(aliases) != {index}:
        missing.append("alias_not_exact_index")
    field = mapping[index]["mappings"]["properties"].get("contextualized_content", {})
    zh = field.get("fields", {}).get("zh", {})
    if (field.get("analyzer", "standard") != "standard" or zh.get("analyzer") != "ik_max_word"
            or zh.get("search_analyzer") != "ik_smart" or settings[index]["settings"]["index"].get("analysis")):
        missing.append("unsupported_lexical_mapping_or_custom_analysis")
    probes = []
    for text in PROBES:
        for analyzer in ("standard", "ik_max_word", "ik_smart"):
            data = (await es.indices.analyze(index=index, analyzer=analyzer, text=text)).body
            tokens = [{k: item[k] for k in ("token", "position", "start_offset", "end_offset")} for item in data["tokens"]]
            probes.append({"text": text, "analyzer": analyzer, "tokens": tokens, "sha256": canonical_hash(tokens)})
    # Node settings can contain secure URLs/custom plugin credentials. Bind their
    # content without publishing it; persist only explicit identity metadata.
    node_metadata = {key: {**{name: value[name] for name in ("name", "version", "roles", "plugins") if name in value},
                           "settings_sha256": canonical_hash(value.get("settings", {}))}
                     for key, value in all_nodes["nodes"].items()}
    content = {"protocol": "lexical-environment-v1", "cluster_uuid": cluster["cluster_uuid"],
               "versions": cluster["version"], "nodes": node_metadata, "mapping": mapping,
               "settings": settings, "aliases": aliases, "node_files": proofs,
               "probe_protocol": "ik-probes-v1", "probes": probes, "implementation": implementation}
    return {"content": content, "environment_sha256": canonical_hash(content), "verified": not missing,
            "missing_evidence": sorted(set(missing)), "observed_at": datetime.now(timezone.utc).isoformat()}


def assert_lexical_environment_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    for value in (before, after):
        if canonical_hash(value["content"]) != value["environment_sha256"]:
            raise ValueError("lexical environment hash drift")
    if before["environment_sha256"] != after["environment_sha256"]:
        raise ValueError("lexical environment drift")
