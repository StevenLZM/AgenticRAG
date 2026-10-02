"""Exercise safe tool-card rendering and persisted conversation restoration."""

from pathlib import Path
import json
import subprocess


def test_tool_cards_dom_contract():
    static = Path(__file__).resolve().parents[3] / "src/agentic_rag/api/static"
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("chat_tool_cards.cjs")), str(static)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-10000:]


def test_real_transit_projection_reaches_collapsible_dom_and_history(tmp_path):
    from agentic_rag.query.tool_answers import build_tool_answer
    from tests.unit.query.test_transit_details import FIXTURE, transit_result

    fixture = json.loads(FIXTURE.read_text())
    answer = build_tool_answer(
        [transit_result(fixture["payload"], arguments=fixture["arguments"])],
        route="fast_rag",
    )
    saved = tmp_path / "answer.json"
    saved.write_text(json.dumps(answer, ensure_ascii=False))
    static = Path(__file__).resolve().parents[3] / "src/agentic_rag/api/static"
    result = subprocess.run(
        [
            "node",
            str(Path(__file__).with_name("chat_transit_cards.cjs")),
            str(static),
            str(saved),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-10000:]
