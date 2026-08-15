"""Opt-in real local MySQL/Redis/Elasticsearch Query pipeline acceptance tests."""

from tests.integration.runtime.test_query_pipeline import (  # noqa: F401
    test_duplicate_delivery_and_worker_restart_keep_one_terminal_run,
    test_query_run_reaches_audited_answer_through_real_boundaries,
)

pytest_plugins = ("tests.fixtures.query_services",)
