"""Stable, sanitized HTTP error responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agentic_rag.persistence.repositories import ActiveRunConflict
from agentic_rag.runtime.ids import new_id


class ApiError(BaseModel):
    """Client-safe error payload shared by every API failure."""

    error_code: str
    message: str
    retryable: bool
    trace_id: str
    degraded_components: tuple[str, ...] = ()


class ApiException(RuntimeError):
    """An expected API failure with an explicitly safe public representation."""

    def __init__(
        self,
        *,
        status_code: int,
        error_code: str,
        message: str,
        retryable: bool = False,
        degraded_components: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.public_message = message
        self.retryable = retryable
        self.degraded_components = degraded_components


class ScopeError(RuntimeError):
    """Raised when the request is outside the caller's permitted scope."""


class RequestFormatError(ValueError):
    """Raised when a request has a valid shape but an invalid format."""


class TemporaryDependencyError(RuntimeError):
    """A provider failure whose private detail must not cross the API boundary."""

    def __init__(self, component: str, *, provider_detail: str | None = None) -> None:
        super().__init__(provider_detail)
        self.component = component


def _response(
    request: Request,
    *,
    status_code: int,
    error_code: str,
    message: str,
    retryable: bool,
    degraded_components: tuple[str, ...] = (),
) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", None) or new_id()
    payload = ApiError(
        error_code=error_code,
        message=message,
        retryable=retryable,
        trace_id=trace_id,
        degraded_components=degraded_components,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


async def _api_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiException)
    return _response(
        request,
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.public_message,
        retryable=exc.retryable,
        degraded_components=exc.degraded_components,
    )


async def _validation_error_handler(request: Request, _exc: Exception) -> JSONResponse:
    return _response(
        request,
        status_code=422,
        error_code="VALIDATION_ERROR",
        message="Request validation failed.",
        retryable=False,
    )


async def _format_error_handler(request: Request, _exc: Exception) -> JSONResponse:
    return _response(
        request,
        status_code=400,
        error_code="INVALID_REQUEST_FORMAT",
        message="The request format is invalid.",
        retryable=False,
    )


async def _scope_error_handler(request: Request, _exc: Exception) -> JSONResponse:
    return _response(
        request,
        status_code=403,
        error_code="SCOPE_FORBIDDEN",
        message="The requested resource is outside the permitted scope.",
        retryable=False,
    )


async def _active_run_handler(request: Request, _exc: Exception) -> JSONResponse:
    response = _response(
        request,
        status_code=409,
        error_code="ACTIVE_RUN_EXISTS",
        message="An active run already exists for this thread.",
        retryable=False,
    )
    existing_run_id = getattr(_exc, "existing_run_id", None)
    if isinstance(existing_run_id, str) and existing_run_id.strip():
        response.headers["Location"] = f"/v1/query-runs/{existing_run_id.strip()}"
    return response


async def _dependency_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, TemporaryDependencyError)
    return _response(
        request,
        status_code=503,
        error_code="DEPENDENCY_UNAVAILABLE",
        message="A required dependency is temporarily unavailable.",
        retryable=True,
        degraded_components=(exc.component,),
    )


async def _unexpected_error_handler(request: Request, _exc: Exception) -> JSONResponse:
    return _response(
        request,
        status_code=500,
        error_code="INTERNAL_ERROR",
        message="An unexpected error occurred.",
        retryable=False,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Install stable mappings without exposing exception strings or tracebacks."""
    app.add_exception_handler(ApiException, _api_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(RequestFormatError, _format_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ScopeError, _scope_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ActiveRunConflict, _active_run_handler)  # type: ignore[arg-type]
    app.add_exception_handler(TemporaryDependencyError, _dependency_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unexpected_error_handler)
