"""Shared pytest configuration."""
import pytest

pytest.register_assert_rewrite("tests.fixtures.query_services", "tests.fixtures.chat_sessions")
