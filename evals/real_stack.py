"""Explicit isolated settings; never mutate the application's .env.local."""
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy.engine import make_url


def isolated_environment(settings, allocation: dict, *, root: Path) -> dict[str, str]:
    suffix = allocation["allocation_id"]
    if not re.fullmatch(r"[a-f0-9]{6,24}", suffix):
        raise ValueError("invalid allocation id")
    required = {"mysql_database": f"agentic_rag_eval_{suffix}",
                "user_id": f"rag-eval-{suffix}", "index_generation": f"eval-{suffix}",
                "mem0_collection": f"agent_memories_eval_{suffix}"}
    if any(allocation[key] != expected for key, expected in required.items()):
        raise ValueError("resources must belong to this evaluation allocation")
    mysql = make_url(settings.mysql_dsn)
    if mysql.database == allocation["mysql_database"]:
        raise ValueError("evaluation database is the business database")
    redis = urlsplit(settings.redis_url)
    db = allocation["redis_database_candidate"]
    if type(db) is not int or not 1 <= db <= 15 or redis.path == f"/{db}":
        raise ValueError("Redis database is not isolated")
    es = urlsplit(allocation["elasticsearch_url"])
    business_es = urlsplit(settings.elasticsearch_url)
    if (es.scheme != "http" or es.hostname not in {"localhost", "127.0.0.1"}
            or es.port in {None, 9200, business_es.port} or es.username or es.password
            or es.path not in {"", "/"} or es.query or es.fragment):
        raise ValueError("requires a dedicated loopback Elasticsearch endpoint")
    artifacts = (root / allocation["artifact_root"]).resolve()
    allowed = (root / "var/artifacts/evals").resolve()
    if not artifacts.is_relative_to(allowed) or artifacts == allowed:
        raise ValueError("artifact root must be inside evaluation artifacts")
    values = {
        # Existing allocations predate IK and must never inherit a business
        # analyzer change into their independently provisioned ES instance.
        "LEXICAL_ANALYSIS": allocation.get("lexical_analysis", "standard"),
        "MYSQL_DSN": mysql.set(database=allocation["mysql_database"]).render_as_string(hide_password=False),
        "REDIS_URL": urlunsplit(redis._replace(path=f"/{db}")),
        "ELASTICSEARCH_URL": allocation["elasticsearch_url"],
        "DEFAULT_USER_ID": allocation["user_id"], "INDEX_GENERATION": allocation["index_generation"],
        "MEM0_COLLECTION": allocation["mem0_collection"], "ARTIFACT_ROOT": str(artifacts),
        "MEM0_HISTORY_DB_PATH": str(artifacts / "mem0/history.db"),
        "QUERY_CHECKPOINT_PATH": str(artifacts / "query.sqlite"),
        "INGESTION_CHECKPOINT_PATH": str(artifacts / "ingestion.sqlite"),
    }
    return {"AGENTIC_RAG_" + key: value for key, value in values.items()}
