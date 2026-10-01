"""Run deterministic browser state and network contracts without a build step."""
from pathlib import Path
import subprocess

import pytest

STATIC = Path(__file__).resolve().parents[3] / "src/agentic_rag/api/static"


@pytest.mark.parametrize("script", ["chat_state.cjs", "chat_transport.cjs"])
def test_chat_client_contract(script):
    result = subprocess.run(["node", str(Path(__file__).with_name(script)), str(STATIC)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.stdout + result.stderr)[-10000:]
