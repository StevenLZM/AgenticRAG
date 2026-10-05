"""Fail-closed durable attempts: never infer a lost result was not sent."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path

from evals.gold_v2_models import canonical_hash
from evals.gold_v2_validation import strict_json
from evals.paired_retrieval_identity import comparison_arms, validate_identity
from evals.report import atomic_write_text

VERIFICATION_KEYS = ("pre_corpus", "post_corpus", "pre_lexical", "post_lexical")


def _write(path, value):
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    path.chmod(0o600)


class PairedLedger:
    @classmethod
    @contextmanager
    def open(cls, root: Path, identity: dict, *, resume: bool):
        validate_identity(identity)
        if root.is_symlink():
            raise ValueError("symlink experiment forbidden")
        if not resume:
            root.mkdir(mode=0o700, parents=True, exist_ok=False)
        elif not root.is_dir():
            raise ValueError("resume experiment missing")
        with (root / ".lock").open("a") as lock:
            (root / ".lock").chmod(0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("experiment already running") from error
            try:
                obj = cls()
                obj.root, obj.identity = root, deepcopy(identity)
                obj._active_phase = False
                obj._current_attempts = []
                if resume:
                    try:
                        saved = strict_json((root / "experiment.json").read_text())
                        obj.data = strict_json((root / "ledger.json").read_text())
                        results = strict_json((root / "results.json").read_text())
                    except (OSError, ValueError) as error:
                        raise ValueError("missing/corrupt experiment ledger") from error
                    if saved != identity:
                        raise ValueError("experiment identity changed")
                    obj._validate()
                    if results != obj.records():
                        raise ValueError("ledger/results mismatch")
                else:
                    obj.data = {"protocol": "paired-ledger-v1", "identity_sha256": identity["sha256"], "events": []}
                    _write(root / "experiment.json", identity)
                    obj._persist()
                yield obj
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _replay(self):
        records, elapsed, phase, previous = {}, 0.0, False, "0" * 64
        valid_cases = {c["case_id"] for c in self.identity["common"]["cases"]}
        valid_arms = comparison_arms(self.identity["common"]["config"])
        seen = set()
        for seq, event in enumerate(self.data["events"], 1):
            if (event["seq"] != seq or event["previous"] != previous or
                    event["sha256"] != canonical_hash({k: v for k, v in event.items() if k != "sha256"})):
                raise ValueError("ledger event chain corrupt")
            previous = event["sha256"]
            value, kind = event["value"], event["kind"]
            if kind == "start":
                key = (value["case_id"], value["arm"])
                if key in seen or key[0] not in valid_cases or key[1] not in valid_arms:
                    raise ValueError("duplicate/foreign attempt")
                seen.add(key)
                if value["attempt_id"] != str(len(records) + 1):
                    raise ValueError("non-monotone attempt")
                records[value["attempt_id"]] = {**value, "status": "uncertain", "elapsed_seconds": None,
                    "identity_sha256": self.identity["sha256"], "interval_verification": "unknown"}
            elif kind == "finish":
                attempt = value["attempt_id"]
                if attempt not in records or records[attempt].get("finished"):
                    raise ValueError("unknown/duplicate finish")
                if value["status"] not in {"success", "failed", "uncertain"}:
                    raise ValueError("unknown outcome")
                seconds = value["elapsed_seconds"]
                if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("invalid elapsed duration")
                if value["status"] == "success" and not isinstance(value.get("hits"), list):
                    raise ValueError("missing successful hits")
                records[attempt].update(value, finished=True)
            elif kind == "phase_start":
                if phase:
                    raise ValueError("unclosed retrieval phase")
                phase = True
            elif kind == "phase_end":
                seconds = value["seconds"]
                if not phase or type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("invalid retrieval phase duration")
                elapsed += seconds
                phase = False
            elif kind == "verification":
                attempts, checks = value["attempt_ids"], value["checks"]
                if (phase or not attempts or len(attempts) != len(set(attempts)) or set(checks) != set(VERIFICATION_KEYS)
                        or any(type(v) is not bool for v in checks.values())):
                    raise ValueError("invalid interval verification")
                for attempt in attempts:
                    if attempt not in records or records[attempt]["interval_verification"] != "unknown":
                        raise ValueError("unknown/duplicate interval verification")
                    records[attempt]["interval_verification"] = "verified" if all(checks.values()) else "failed"
                    records[attempt]["interval_checks"] = deepcopy(checks)
            else:
                raise ValueError("unknown ledger event")
        if len(records) > self.identity["common"]["config"]["request_limit"]:
            raise ValueError("request budget exceeded")
        return list(records.values()), elapsed, phase

    def _validate(self):
        if self.data.get("protocol") != "paired-ledger-v1" or self.data.get("identity_sha256") != self.identity["sha256"]:
            raise ValueError("ledger binding mismatch")
        try:
            self._replay()
        except (KeyError, TypeError) as error:
            raise ValueError("malformed ledger") from error

    def _persist(self):
        self._validate()
        _write(self.root / "ledger.json", self.data)
        _write(self.root / "results.json", self.records())

    def _append(self, kind, value):
        events = self.data["events"]
        event = {"seq": len(events) + 1, "previous": events[-1]["sha256"] if events else "0" * 64,
                 "kind": kind, "value": deepcopy(value)}
        self.data["events"].append({**event, "sha256": canonical_hash(event)})
        try:
            self._persist()
        except BaseException:
            self.data["events"].pop()
            raise

    def records(self):
        return deepcopy(self._replay()[0])

    @property
    def remaining_wall(self):
        _, elapsed, phase = self._replay()
        return 0.0 if phase else max(0.0, self.identity["common"]["config"]["wall_s"] - elapsed)

    def begin_phase(self):
        if self.remaining_wall <= 0:
            raise ValueError("wall budget exhausted or previous phase uncertain")
        self._append("phase_start", {})
        self._active_phase = True

    def checkpoint_elapsed(self, seconds: float):
        self._append("phase_end", {"seconds": seconds})
        self._active_phase = False

    def start(self, request):
        if not self._active_phase and self.remaining_wall <= 0:
            raise ValueError("wall budget exhausted or previous phase uncertain")
        records = self.records()
        if len(records) >= self.identity["common"]["config"]["request_limit"]:
            raise ValueError("request budget exhausted")
        attempt = str(len(records) + 1)
        self._append("start", {**request, "attempt_id": attempt})
        self._current_attempts.append(attempt)
        return attempt

    def finish(self, attempt_id, outcome):
        if set(outcome) - {"status", "hits", "elapsed_seconds", "error_type", "took_ms", "response_sha256"}:
            raise ValueError("unexpected outcome fields")
        self._append("finish", {**outcome, "attempt_id": attempt_id})

    def attest_current_interval(self, checks):
        """Only attest requests made by this process; past unknowns stay unknown."""
        if self._current_attempts:
            self._append("verification", {"attempt_ids": self._current_attempts,
                "checks": {key: checks.get(key) is True for key in VERIFICATION_KEYS}})
            self._current_attempts = []
