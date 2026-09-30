"""Strict Pydantic models and JSONL validation for offline evaluation data."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StringConstraints, field_validator, model_validator
from typing_extensions import Annotated


NonEmptyText = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)
]
SafeIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$",
    ),
]

_DATASET_MINIMUMS = {"baseline": 24, "ingestion_fidelity": 12, "security": 12}
_SECRET_TEXT = re.compile(
    r"(?:api[_ -]?key|authorization:\s*bearer|password\s*=|sk-[A-Za-z0-9_-]{8,}|"
    r"chain[- ]of[- ]thought|hidden reasoning|provider[_ -]?secret)",
    re.IGNORECASE,
)
_UNSAFE_KEY = re.compile(
    r"(?:prompt|completion|raw[_ -]?tool|tool[_ -]?(?:input|output|args|result)|"
    r"chain[_ -]?of[_ -]?thought|hidden[_ -]?reasoning|provider[_ -]?(?:token|response|secret))",
    re.IGNORECASE,
)


class EvaluationCase(BaseModel):
    """One fixed, reviewable evaluation case.

    ``extra='forbid'`` is intentional: adding a field to a dataset requires a
    reviewed schema change rather than silently changing the evaluation input.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: SafeIdentifier
    user_id: NonEmptyText = "eval_user"
    question: Annotated[
        str,
        StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=8_000),
    ]
    answerable: StrictBool = True
    reference_answer: Annotated[
        str,
        StringConstraints(strict=True, strip_whitespace=True, max_length=20_000),
    ]
    reference_parent_ids: tuple[SafeIdentifier, ...] = Field(max_length=128)
    expected_route: Literal["fast_rag", "research"]
    tags: tuple[SafeIdentifier, ...] = Field(min_length=1, max_length=32)
    runtime_config_snapshot_id: SafeIdentifier

    @model_validator(mode="after")
    def _answerability_matches_gold(self) -> EvaluationCase:
        if self.answerable:
            if not self.reference_answer or not self.reference_parent_ids:
                raise ValueError("answerable cases require a reference answer and parent IDs")
        elif self.reference_parent_ids:
            raise ValueError("unanswerable cases must not have gold parent IDs")
        return self

    @field_validator("reference_parent_ids")
    @classmethod
    def _unique_references(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("reference_parent_ids must be unique")
        return values

    @field_validator("tags")
    @classmethod
    def _unique_tags(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("tags must be unique")
        return values

    @field_validator("question", "reference_answer")
    @classmethod
    def _no_provider_secrets(cls, value: str) -> str:
        if _SECRET_TEXT.search(value):
            raise ValueError("evaluation text contains provider secret or hidden reasoning")
        return value


class IngestionFidelityCase(EvaluationCase):
    """Case with deterministic expected parser/chunk provenance locators."""

    fixture_path: Annotated[
        str,
        StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=512),
    ]
    expected_ast_locators: tuple[NonEmptyText, ...] = Field(min_length=1, max_length=64)
    expected_content_types: tuple[Literal["text", "table", "ocr", "spreadsheet"], ...] = Field(
        min_length=1, max_length=16
    )

    @field_validator("fixture_path", mode="before")
    @classmethod
    def _relative_fixture_path(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("fixture_path must be a string")
        candidate = value.strip()
        posix_path = PurePosixPath(candidate)
        windows_path = PureWindowsPath(candidate)
        if (
            not candidate
            or "\\" in candidate
            or posix_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or candidate.startswith(("/", "~"))
            or ".." in posix_path.parts
        ):
            raise ValueError("fixture_path must be a repository-relative path")
        return candidate


class SecurityCase(EvaluationCase):
    """Adversarial input case with an expected deterministic safety outcome."""

    security_scenario: Literal[
        "cross_user_query",
        "prompt_injection",
        "hidden_unicode",
        "forged_evidence",
        "memory_instruction",
        "filter_override",
    ]
    expected_user_leak_count: StrictInt = Field(default=0, ge=0)
    expected_security_outcome: Literal["blocked", "sanitized", "scoped"]


DatasetCase: TypeAlias = EvaluationCase | IngestionFidelityCase | SecurityCase


def load_jsonl_dataset(path: str | Path, *, dataset_name: str | None = None) -> list[DatasetCase]:
    """Load and strictly validate one JSONL dataset.

    JSON constants such as ``NaN`` and ``Infinity`` are rejected before Pydantic
    sees them.  This is stricter than the permissive default of ``json.loads``.
    """

    dataset_path = Path(path)
    name = dataset_name or _dataset_name_from_path(dataset_path)
    rows: list[Mapping[str, Any]] = []
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"unable to read dataset {dataset_path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{dataset_path}:{line_number}: blank JSONL row")
        try:
            value = json.loads(line, parse_constant=_reject_nonfinite)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{dataset_path}:{line_number}: invalid JSON or non-finite value") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"{dataset_path}:{line_number}: row must be a JSON object")
        _assert_json_safe(value, path=f"{dataset_path}:{line_number}")
        rows.append(value)
    return validate_dataset_rows(rows, dataset_name=name)


def validate_dataset_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_name: str,
    enforce_minimum: bool = False,
) -> list[DatasetCase]:
    """Validate rows for ``baseline``, ``ingestion_fidelity`` or ``security``.

    Duplicate IDs are rejected within a dataset.  The directory-level validator
    additionally enforces the required minimum counts and global ID uniqueness.
    """

    normalized_name = _normalize_dataset_name(dataset_name)
    model: type[DatasetCase]
    if normalized_name == "baseline":
        model = EvaluationCase
    elif normalized_name == "ingestion_fidelity":
        model = IngestionFidelityCase
    else:
        model = SecurityCase
    result: list[DatasetCase] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{normalized_name}:{index}: row must be an object")
        _assert_json_safe(row, path=f"{normalized_name}:{index}")
        try:
            case = model.model_validate(dict(row))
        except Exception as error:
            raise ValueError(f"{normalized_name}:{index}: invalid case: {error}") from error
        if case.case_id in seen:
            raise ValueError(f"{normalized_name}:{index}: duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        _assert_no_unsafe_keys(row, path=f"{normalized_name}:{index}")
        result.append(case)
    if enforce_minimum and len(result) < _DATASET_MINIMUMS[normalized_name]:
        raise ValueError(
            f"{normalized_name}: expected at least {_DATASET_MINIMUMS[normalized_name]} cases, got {len(result)}"
        )
    return result


def validate_dataset_directory(path: str | Path) -> dict[str, int]:
    """Validate all fixed datasets and return their case counts."""

    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"dataset directory does not exist: {root}")
    counts: dict[str, int] = {}
    all_ids: set[str] = set()
    for name in ("baseline", "ingestion_fidelity", "security"):
        dataset_path = root / f"{name}.jsonl"
        cases = load_jsonl_dataset(dataset_path, dataset_name=name)
        if len(cases) < _DATASET_MINIMUMS[name]:
            raise ValueError(
                f"{dataset_path}: expected at least {_DATASET_MINIMUMS[name]} cases, got {len(cases)}"
            )
        duplicate = all_ids.intersection(case.case_id for case in cases)
        if duplicate:
            raise ValueError(f"duplicate case_id across datasets: {sorted(duplicate)[0]}")
        all_ids.update(case.case_id for case in cases)
        counts[name] = len(cases)
    return counts


def _normalize_dataset_name(value: str) -> Literal["baseline", "ingestion_fidelity", "security"]:
    normalized = value.removesuffix(".jsonl").strip().casefold()
    if normalized not in _DATASET_MINIMUMS:
        raise ValueError(f"unknown dataset name {value!r}")
    return normalized  # type: ignore[return-value]


def _dataset_name_from_path(path: Path) -> str:
    try:
        return _normalize_dataset_name(path.stem)
    except ValueError as error:
        raise ValueError("dataset name must be baseline, ingestion_fidelity, or security") from error


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


def _assert_json_safe(value: object, *, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: non-finite numeric value")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: JSON object keys must be strings")
            _assert_json_safe(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _assert_json_safe(nested, path=f"{path}[{index}]")
    elif value is None or isinstance(value, (str, int, bool, float)):
        return
    else:
        raise ValueError(f"{path}: value is not JSON-compatible")


def _assert_no_unsafe_keys(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and _UNSAFE_KEY.search(key):
                raise ValueError(f"{path}: unsafe telemetry/provider field {key!r}")
            _assert_no_unsafe_keys(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _assert_no_unsafe_keys(nested, path=f"{path}[{index}]")
