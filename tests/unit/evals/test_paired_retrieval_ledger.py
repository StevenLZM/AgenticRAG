import json

import pytest

from tests.unit.evals.test_paired_retrieval_identity import identity_fixture


def test_crash_after_send_is_uncertain_without_retry(tmp_path):
    from evals.paired_retrieval_ledger import PairedLedger
    case, identity = identity_fixture()
    with PairedLedger.open(tmp_path / "run", identity, resume=False) as ledger:
        ledger.begin_phase()
        attempt = ledger.start({"case_id": case.case_id, "arm": "standard", "request": {"size": 40}})
    with PairedLedger.open(tmp_path / "run", identity, resume=True) as resumed:
        assert resumed.records()[0]["status"] == "uncertain"
        assert resumed.records()[0]["attempt_id"] == attempt
        assert resumed.remaining_wall == 0
        with pytest.raises(ValueError):
            resumed.start({"case_id": case.case_id, "arm": "standard", "request": {"size": 40}})


@pytest.mark.parametrize("fault", ["missing", "hash", "negative", "unknown_finish"])
def test_missing_or_tampered_ledger_refused(tmp_path, fault):
    from evals.paired_retrieval_ledger import PairedLedger
    case, identity = identity_fixture()
    path = tmp_path / "run"
    with PairedLedger.open(path, identity, resume=False) as ledger:
        ledger.start({"case_id": case.case_id, "arm": "standard", "request": {"size": 40}})
    data = path / "ledger.json"
    if fault == "missing":
        data.unlink()
    else:
        raw = json.loads(data.read_text())
        raw["events"][0]["kind"] = fault
        data.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        with PairedLedger.open(path, identity, resume=True):
            pass


def test_double_finish_and_concurrent_writer_refused(tmp_path):
    from evals.paired_retrieval_ledger import PairedLedger
    case, identity = identity_fixture()
    with PairedLedger.open(tmp_path / "run", identity, resume=False) as ledger:
        with pytest.raises(ValueError):
            with PairedLedger.open(tmp_path / "run", identity, resume=True):
                pass
        attempt = ledger.start({"case_id": case.case_id, "arm": "standard", "request": {"size": 40}})
        ledger.finish(attempt, {"status": "failed", "elapsed_seconds": 0.1, "error_type": "error"})
        with pytest.raises(ValueError):
            ledger.finish(attempt, {"status": "failed", "elapsed_seconds": 0.1})


def test_crashed_phase_blocks_other_pending_requests(tmp_path):
    from evals.paired_retrieval_ledger import PairedLedger
    case, identity = identity_fixture()
    with PairedLedger.open(tmp_path / "run", identity, resume=False) as ledger:
        ledger.begin_phase()
        ledger.start({"case_id": case.case_id, "arm": "standard", "request": {"size": 40}})
    with PairedLedger.open(tmp_path / "run", identity, resume=True) as ledger:
        with pytest.raises(ValueError, match="wall|uncertain"):
            ledger.start({"case_id": case.case_id, "arm": "ik", "request": {"size": 40}})


def test_valid_json_truncation_cannot_discard_attempts(tmp_path):
    from evals.paired_retrieval_ledger import PairedLedger
    case, identity = identity_fixture()
    with PairedLedger.open(tmp_path / "run", identity, resume=False) as ledger:
        ledger.start({"case_id": case.case_id, "arm": "standard", "request": {"size": 40}})
    path = tmp_path / "run/ledger.json"
    raw = json.loads(path.read_text())
    raw["events"] = []
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="results"):
        with PairedLedger.open(tmp_path / "run", identity, resume=True):
            pass
