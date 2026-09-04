"""The local-only Calc Flow Studio API factory and server entry points."""

from __future__ import annotations

import ipaddress
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from calc_flow import FileProjectStore, Runtime
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from calc_flow_studio.routes import (
    API_PREFIX,
    ProjectStoreProtocol,
    RunManagerProtocol,
    RuntimeProtocol,
    register_capability_routes,
    register_job_routes,
    register_project_routes,
)
from calc_flow_studio.run_manager import RunManager
from calc_flow_studio.web_errors import (
    MAX_PROJECT_IMPORT_BYTES,
    join_validation_error_detail,
)

__all__ = [
    "API_PREFIX",
    "MAX_PROJECT_IMPORT_BYTES",
    "ProjectStoreProtocol",
    "RunManagerProtocol",
    "RuntimeProtocol",
    "create_app",
    "serve",
    "validate_bind_host",
]


def _default_frontend_directory() -> Path | None:
    static = Path(__file__).with_name("static")
    return static if (static / "index.html").is_file() else None


def create_app(
    *,
    project_directory: str | Path = ".calc-flow-projects",
    checkpoint_directory: str | Path = ".calc-flow-checkpoints",
    project_store: ProjectStoreProtocol | None = None,
    runtime: RuntimeProtocol | None = None,
    run_manager: RunManagerProtocol | None = None,
    frontend_directory: str | Path | None = None,
) -> FastAPI:
    """Create the local-only v3 API without opening a network listener."""
    projects = project_store or FileProjectStore(project_directory)
    selected_runtime = runtime or Runtime()
    selected_run_manager = (
        run_manager
        if run_manager is not None
        else RunManager(
            runtime=selected_runtime if isinstance(selected_runtime, Runtime) else None,
            checkpoint_directory=checkpoint_directory,
        )
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await run_in_threadpool(selected_run_manager.shutdown)

    app = FastAPI(title="Calc Flow API", version="4.0.0", lifespan=lifespan)
    app.state.project_store = projects
    app.state.runtime = selected_runtime
    app.state.run_manager = selected_run_manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @app.exception_handler(RequestValidationError)
    async def join_raw_validation_envelope(
        _: Request, error: RequestValidationError
    ) -> JSONResponse:
        detail = join_validation_error_detail(error)
        if detail is not None:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": detail},
            )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": jsonable_encoder(error.errors())},
        )

    register_capability_routes(app, selected_runtime, selected_run_manager)
    register_project_routes(app, projects, selected_runtime)
    register_job_routes(app, projects, selected_run_manager)

    frontend = (
        Path(frontend_directory)
        if frontend_directory is not None
        else _default_frontend_directory()
    )
    if frontend is not None:
        assets = frontend / "assets"
        index = frontend / "index.html"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")
        if index.is_file():

            @app.get("/", include_in_schema=False)
            def frontend_index() -> FileResponse:
                return FileResponse(index)

    return app


def validate_bind_host(host: str) -> str:
    if host == "localhost":
        return host
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        message = "Calc Flow web server host must be a loopback IP or localhost"
        raise ValueError(message) from error
    if not address.is_loopback:
        message = "Calc Flow web server may bind only to a loopback address"
        raise ValueError(message)
    return host


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    project_directory: str | Path = ".calc-flow-projects",
    checkpoint_directory: str | Path = ".calc-flow-checkpoints",
) -> None:
    """Run the unauthenticated v3 service on a loopback interface only."""
    import uvicorn

    validate_bind_host(host)
    uvicorn.run(
        create_app(
            project_directory=project_directory,
            checkpoint_directory=checkpoint_directory,
        ),
        host=host,
        port=port,
    )
