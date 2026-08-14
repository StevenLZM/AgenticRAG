"""Real-service Query API -> Outbox -> Worker -> Graph integration contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest_plugins = ("tests.fixtures.query_services",)

if TYPE_CHECKING:
    from tests.fixtures.query_services import RealQueryFixture

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


async def test_query_run_reaches_audited_answer_through_real_boundaries(
    real_query_fixture: RealQueryFixture,
) -> None:
    response = await real_query_fixture.create_query(
        "What does the seeded production document require?"
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    terminal = await real_query_fixture.wait_for_terminal(run_id)

    assert terminal["status"] == "completed"
    assert terminal["answer"]["audited"] is True
    assert terminal["answer"]["segments"]
    assert await real_query_fixture.events_contain(run_id, "ANSWER_FINALIZED")
    assert "ANSWER_FINALIZED" in await real_query_fixture.sse_body(run_id)


async def test_duplicate_delivery_and_worker_restart_keep_one_terminal_run(
    real_query_fixture: RealQueryFixture,
) -> None:
    response = await real_query_fixture.create_query(
        "What does the seeded production document require?", thread_id="restart"
    )
    run_id = response.json()["run_id"]
    await real_query_fixture.inject_duplicate_delivery(run_id)
    terminal = await real_query_fixture.wait_for_terminal(run_id, restart_worker=True)

    assert terminal["status"] == "completed"
    assert await real_query_fixture.terminal_event_count(run_id) == 1
