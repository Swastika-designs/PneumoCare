"""Application error types translated into consistent JSON error responses."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationFailed(AppError):
    status_code = 422
    code = "validation_failed"


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class DatasetUnavailable(AppError):
    """A dataset needed for this analysis has not been downloaded."""

    status_code = 409
    code = "dataset_unavailable"


class ModelUnavailable(AppError):
    """A model checkpoint has not been trained yet."""

    status_code = 409
    code = "model_unavailable"


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle(_request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_request: Request, exc: StarletteHTTPException):
        # Unmatched routes / methods use the same envelope as application errors.
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(status_code=exc.status_code,
                            content={"error": {"code": code, "message": str(exc.detail), "details": {}}})

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(_request: Request, exc: RequestValidationError):
        problems = [f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": "; ".join(problems),
                               "details": {"errors": jsonable_encoder(exc.errors())}}},
        )
