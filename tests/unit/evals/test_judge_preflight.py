import json

import pytest

from evals.ragas_adapter import RagasEvaluation
from scripts.preflight_ragas import run_preflight


class Judge:
    fingerprint = "contract-only"
    metadata = {"judge_model": "contract"}
    calls = 0

    async def evaluate(self, **kwargs):
        self.calls += 1
        return RagasEvaluation("available", {"correct_refusal": 1.0} if not kwargs["answerable"]
                               else {"faithfulness": 0.9})


@pytest.mark.asyncio
async def test_preflight_persists_and_does_not_repeat_completed_calls(tmp_path):
    judge, path = Judge(), tmp_path / "preflight.json"
    first = await run_preflight(judge, path)
    assert first["purpose"] == "judge-connectivity-only"
    assert first["real_query_count"] == 0
    assert first["status"] == "available"
    assert json.loads(path.read_text())["status"] == "available"
    await run_preflight(judge, path)
    assert judge.calls == 2


@pytest.mark.asyncio
async def test_preflight_does_not_silently_repeat_uncertain_calls(tmp_path):
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps({"judge_fingerprint": "contract-only", "status": "running"}))
    judge = Judge()
    with pytest.raises(ValueError, match="retry"):
        await run_preflight(judge, path)
    assert judge.calls == 0
