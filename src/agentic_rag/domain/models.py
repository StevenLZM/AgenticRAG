"""Shared domain value objects and status protocols."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


class RunStatus(StrEnum):
    """Lifecycle states for a query run."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(StrEnum):
    """Lifecycle states for an ingestion job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    QUARANTINED = "quarantined"
    FAILED = "failed"


class DocumentStatus(StrEnum):
    """Lifecycle states for a document."""

    PROCESSING = "processing"
    ACTIVE = "active"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentVersionStatus(StrEnum):
    """Lifecycle states for a version of a document."""

    UPLOADED = "uploaded"
    BUILDING = "building"
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    FAILED = "failed"
    INACTIVE = "inactive"


class UserScope(BaseModel):
    """An immutable, normalized boundary for user-owned data."""

    model_config = ConfigDict(frozen=True)

    user_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
