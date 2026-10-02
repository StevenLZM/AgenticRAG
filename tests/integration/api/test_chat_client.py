"""Run deterministic browser state and network contracts without a build step."""

from pathlib import Path
import subprocess

import pytest

STATIC = Path(__file__).resolve().parents[3] / "src/agentic_rag/api/static"


@pytest.mark.parametrize(
    "script",
    [
        "chat_state.cjs",
        "chat_tools.cjs",
        "chat_transport.cjs",
        "chat_view_flow.cjs",
        "chat_recovery_flow.cjs",
    ],
)
def test_chat_client_contract(script):
    result = subprocess.run(
        ["node", str(Path(__file__).with_name(script)), str(STATIC)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-10000:]


@pytest.mark.parametrize(
    "case",
    [
        "lateCancel",
        "crossSessionCancel",
        "historySwitch",
        "deletedCreation",
        "lateSubmissionCheck",
        "lateSubmissionError",
        "deleteNavigation",
        "historyGap",
        "revokedObservation",
        "foregroundDuringSubmit",
        "overlappingForegroundRefresh",
        "acknowledgedAfterReopen",
        "missedTurnCatchup",
        "revokedSubmission",
    ],
)
def test_chat_controller_lifecycle(case):
    result = subprocess.run(
        [
            "node",
            str(Path(__file__).with_name("chat_lifecycle_flow.cjs")),
            str(STATIC),
            case,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-10000:]
