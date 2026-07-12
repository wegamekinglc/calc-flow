from __future__ import annotations

import ipaddress
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from calc_flow.checkpoint import CheckpointError, FileCheckpointStore
from calc_flow.config import (
    CONFIG_FORMAT_VERSION,
    MAX_PREVIEW_BYTES,
    MAX_PREVIEW_ROWS,
    MAX_PREVIEW_SECONDS,
    SUPPORTED_ARROW_TYPES,
    ProjectConfig,
    RunOptions,
    ValidationReport,
    compile_project,
    project_json_schema,
    validate_project,
)
from calc_flow.project_store import (
    FileProjectStore,
    ProjectConflictError,
    ProjectFormatError,
    ProjectNotFoundError,
    export_project_document,
    load_project_document,
)
from calc_flow.udf import UdfRegistry, UdfRegistrySnapshot
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from calc_flow_studio.models import (
    CatalogResponse,
    CheckpointSummary,
    ProjectCreateRequest,
    ProjectSummary,
    RunRequest,
    RunResponse,
    RunStatus,
)
from calc_flow_studio.run_manager import RunManager, RunManagerError


def _project_summary(project: ProjectConfig) -> ProjectSummary:
    return ProjectSummary(
        id=project.id,
        name=project.name,
        description=project.description,
        node_count=len(project.pipeline.nodes),
    )


def _not_found(error: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))


def _default_frontend_directory() -> Path | None:
    static = Path(__file__).with_name("static")
    return static if (static / "index.html").is_file() else None


def _import_project_document(
    store: FileProjectStore,
    content: bytes,
    *,
    format: str,
    replace: bool,
) -> ProjectConfig:
    project = load_project_document(content, format=format)
    if replace:
        store.put(project)
    else:
        store.create(project)
    return project


def create_app(
    *,
    project_directory: str | Path = ".calc-flow-projects",
    checkpoint_directory: str | Path = ".calc-flow-checkpoints",
    udf_registry: UdfRegistry | UdfRegistrySnapshot | None = None,
    run_manager: RunManager | None = None,
    frontend_directory: str | Path | None = None,
) -> FastAPI:
    """Create the local-only API without opening a network listener."""
    store = FileProjectStore(project_directory)
    checkpoint_store = FileCheckpointStore(checkpoint_directory)
    manager = run_manager or RunManager(udf_registry=udf_registry)
    registry = manager.udf_registry

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        manager.shutdown()

    app = FastAPI(
        title="Calc Flow API",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.project_store = store
    app.state.checkpoint_store = checkpoint_store
    app.state.run_manager = manager
    app.state.udf_registry = registry
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/v1/catalog", response_model=CatalogResponse)
    def get_catalog() -> CatalogResponse:
        defaults = RunOptions()
        return CatalogResponse(
            config_format_version=CONFIG_FORMAT_VERSION,
            operators=(
                {
                    "kind": "expression",
                    "label": "DataFusion expression",
                    "backend_selector": False,
                    "capabilities": ["calculation", "projection", "filter"],
                },
                {
                    "kind": "sql",
                    "label": "DataFusion SQL",
                    "backend_selector": False,
                    "capabilities": ["join", "aggregate", "window", "sort"],
                },
                {
                    "kind": "array_expression",
                    "label": "Array expression",
                    "backends": ["numpy", "jax"],
                },
            ),
            udfs=registry.catalog(),
            arrow_types=SUPPORTED_ARROW_TYPES,
            limits={
                "max_input_bytes": MAX_PREVIEW_BYTES,
                "max_rows": MAX_PREVIEW_ROWS,
                "max_seconds": MAX_PREVIEW_SECONDS,
                "memory_limit_mb": defaults.memory_limit_mb,
                "output_rows": defaults.output_rows,
            },
        )

    @app.get("/api/v1/schema/project")
    def get_project_schema() -> dict:
        return dict(project_json_schema())

    @app.get("/api/v1/projects", response_model=tuple[ProjectSummary, ...])
    def list_projects() -> tuple[ProjectSummary, ...]:
        return tuple(_project_summary(project) for project in store.list())

    @app.post(
        "/api/v1/projects",
        response_model=ProjectConfig,
        status_code=status.HTTP_201_CREATED,
    )
    def create_project(request: ProjectCreateRequest) -> ProjectConfig:
        for _ in range(3):
            project = request.to_project(f"project_{uuid4().hex}")
            try:
                store.create(project)
            except ProjectConflictError:
                continue
            return project
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="could not allocate a unique project ID",
        )

    @app.post(
        "/api/v1/projects/import",
        response_model=ProjectConfig,
        status_code=status.HTTP_201_CREATED,
    )
    async def import_project(
        request: Request,
        format: str = Query(pattern="^(json|yaml)$"),
        replace: bool = False,
    ) -> ProjectConfig:
        content = await request.body()
        try:
            project = await run_in_threadpool(
                _import_project_document,
                store,
                content,
                format=format,
                replace=replace,
            )
        except ProjectConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(error)
            ) from error
        except ProjectFormatError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        return project

    @app.get("/api/v1/projects/{project_id}", response_model=ProjectConfig)
    def get_project(project_id: str) -> ProjectConfig:
        try:
            return store.get(project_id)
        except ProjectNotFoundError as error:
            raise _not_found(error) from error

    @app.put("/api/v1/projects/{project_id}", response_model=ProjectConfig)
    def put_project(project_id: str, project: ProjectConfig) -> ProjectConfig:
        if project.id != project_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="path project ID does not match the document",
            )
        store.put(project)
        return project

    @app.delete("/api/v1/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_project(project_id: str) -> Response:
        try:
            store.delete(project_id)
        except ProjectNotFoundError as error:
            raise _not_found(error) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/v1/projects/{project_id}/export")
    def export_project(
        project_id: str,
        format: str = Query(default="json", pattern="^(json|yaml)$"),
    ) -> PlainTextResponse:
        try:
            project = store.get(project_id)
            document = export_project_document(project, format=format)
        except ProjectNotFoundError as error:
            raise _not_found(error) from error
        except ProjectFormatError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        media_type = "application/json" if format == "json" else "application/yaml"
        return PlainTextResponse(
            document,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{project_id}.{format}"'
            },
        )

    @app.post(
        "/api/v1/projects/{project_id}/validate",
        response_model=ValidationReport,
    )
    def validate_stored_project(project_id: str) -> ValidationReport:
        try:
            project = store.get(project_id)
        except ProjectNotFoundError as error:
            raise _not_found(error) from error
        return validate_project(project, udf_registry=registry)

    def checkpoint_summary(project: ProjectConfig) -> CheckpointSummary:
        try:
            plan = compile_project(project, udf_registry=registry)
            checkpoint = checkpoint_store.load(plan.name)
        except CheckpointError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"project cannot be compiled: {error}",
            ) from error
        if checkpoint is None:
            return CheckpointSummary(pipeline_name=plan.name, exists=False)
        return CheckpointSummary(
            pipeline_name=plan.name,
            exists=True,
            compatible=checkpoint.pipeline_fingerprint == plan.fingerprint,
            pipeline_fingerprint=checkpoint.pipeline_fingerprint,
            sequence=checkpoint.sequence,
            source_cursor=checkpoint.source_cursor,
            created_at=checkpoint.created_at,
            state_nodes=tuple(sorted(checkpoint.state)),
        )

    @app.get(
        "/api/v1/projects/{project_id}/checkpoint",
        response_model=CheckpointSummary,
    )
    def get_project_checkpoint(project_id: str) -> CheckpointSummary:
        try:
            project = store.get(project_id)
        except ProjectNotFoundError as error:
            raise _not_found(error) from error
        return checkpoint_summary(project)

    @app.delete(
        "/api/v1/projects/{project_id}/checkpoint",
        response_model=CheckpointSummary,
    )
    def reset_project_checkpoint(project_id: str) -> CheckpointSummary:
        try:
            project = store.get(project_id)
        except ProjectNotFoundError as error:
            raise _not_found(error) from error
        checkpoint_store.delete(project.pipeline.name)
        return CheckpointSummary(
            pipeline_name=project.pipeline.name,
            exists=False,
        )

    @app.post(
        "/api/v1/projects/{project_id}/runs",
        response_model=RunResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_run(project_id: str, request: RunRequest) -> RunResponse:
        try:
            project = store.get(project_id)
            return manager.submit(project, request)
        except ProjectNotFoundError as error:
            raise _not_found(error) from error
        except RunManagerError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

    @app.get("/api/v1/runs/{run_id}", response_model=RunResponse)
    def get_run(run_id: str) -> RunResponse:
        try:
            return manager.get(run_id)
        except KeyError as error:
            raise _not_found(error) from error

    @app.get("/api/v1/runs/{run_id}/events")
    def get_run_events(
        run_id: str,
        last_event_id: int | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        try:
            manager.get(run_id)
        except KeyError as error:
            raise _not_found(error) from error

        def stream() -> Iterator[str]:
            after_sequence = last_event_id if last_event_id is not None else -1
            terminal = {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.TIMED_OUT,
            }
            yield "retry: 500\n\n"
            while True:
                events, run_status = manager.wait_for_events(
                    run_id,
                    after_sequence=after_sequence,
                    timeout=10.0,
                )
                if not events:
                    if run_status in terminal:
                        return
                    yield ": keep-alive\n\n"
                    continue
                for event in events:
                    payload = json.dumps(
                        event.model_dump(mode="json"), separators=(",", ":")
                    )
                    yield (
                        f"id: {event.sequence}\nevent: {event.type}\n"
                        f"data: {payload}\n\n"
                    )
                    after_sequence = event.sequence
                if run_status in terminal:
                    return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.delete("/api/v1/runs/{run_id}", response_model=RunResponse)
    def cancel_run(run_id: str) -> RunResponse:
        try:
            return manager.cancel(run_id)
        except KeyError as error:
            raise _not_found(error) from error

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
        msg = "Calc Flow web server host must be a loopback IP or localhost"
        raise ValueError(msg) from error
    if not address.is_loopback:
        msg = "Calc Flow web server may bind only to a loopback address"
        raise ValueError(msg)
    return host


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    project_directory: str | Path = ".calc-flow-projects",
    checkpoint_directory: str | Path = ".calc-flow-checkpoints",
) -> None:
    """Run the unauthenticated v0.2 service on a loopback interface only."""
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
