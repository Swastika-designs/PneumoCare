"""FastAPI application entry point.

Run from the ``backend`` folder:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analytics, ehr, imaging, operations, reports, screening, system, text
from app.core.config import settings
from app.core.errors import register_error_handlers
from app.db import database

logger = logging.getLogger("pneumocare")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_dirs()
    if database.engine is None:
        database.configure()
    database.init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan,
                  description="Healthcare analytics course project: pneumonia detection + 15 tutorial modules.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    register_error_handlers(app)

    @app.exception_handler(Exception)
    async def _unexpected(_request: Request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error",
                                                                "message": f"Unexpected server error: {exc}",
                                                                "details": {}}})

    for module in (system, screening, ehr, imaging, text, operations, analytics, reports):
        app.include_router(module.router)
    return app


app = create_app()
