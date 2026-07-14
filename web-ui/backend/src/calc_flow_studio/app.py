from __future__ import annotations

import ipaddress
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from calc_flow import (
    CalcFlowError,
    FileCheckpointStore,
    FileProjectStore,
    ProjectDocument,
    Runtime,
    project_json_schema,
)
from calc_flow.store import (
    export_project_json,
    export_project_yaml,
    import_project_json,
    import_project_yaml,
)
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from calc_flow_studio.models import (
    CheckpointSummary,
    ProjectCreateRequest,
    ProjectSummary,
    RunEvent,
    RunRequest,
    RunResponse,
    RunStatus,
)

API_PREFIX = "/api/v2"


class ProjectStoreProtocol(Protocol):
    async def create(self, project: ProjectDocument) -> None: ...

    async def put(self, project: ProjectDocument) -> None: ...

    async def get(self, project_id: str) -> ProjectDocument: ...

    async def list(self) -> list[ProjectDocument]: ...

    async def delete(self, project_id: str) -> None: ...


class CheckpointStoreProtocol(Protocol):
    async def load(self, pipeline_name: str) -> dict[str, object] | None: ...

    async def delete(self, pipeline_name: str) -> None: ...


class RuntimeProtocol(Protocol):
    def catalog(self) -> list[dict[str, object]]: ...

    def validation_report(self, project_json: str) -> dict[str, object]: ...

    def compile_project(self, project_json: str) -> object: ...


class RunManagerProtocol(Protocol):
    def submit(self, project: ProjectDocument, request: RunRequest) -> RunResponse: ...

    def get(self, run_id: str) -> RunResponse: ...

    def wait_for_events(
        self, run_id: str, *, after_sequence: int, timeout: float
    ) -> tuple[tuple[RunEvent, ...], RunStatus]: ...

    def cancel(self, run_id: str) -> RunResponse: ...

    def shutdown(self) -> None: ...


def _project_summary(project: ProjectDocument) -> ProjectSummary:
    root = project.root
    pipeline = root["pipeline"]
    assert isinstance(pipeline, dict)
    nodes = pipeline["nodes"]
    assert isinstance(nodes, list)
    return ProjectSummary(
        id=str(root["id"]),
        name=str(root["name"]),
        description=str(root["description"]),
        node_count=len(nodes),
    )


def _http_error(code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=code, detail=detail)


def _native_error(error: Exception, *, operation: str) -> HTTPException:
    message = str(error)
    if operation in {"get", "delete"} and (
        isinstance(error, KeyError) or "not found" in message
    ):
        return _http_error(status.HTTP_404_NOT_FOUND, message)
    if operation == "create" and "already exists" in message:
        return _http_error(status.HTTP_409_CONFLICT, message)
    return _http_error(status.HTTP_422_UNPROCESSABLE_CONTENT, message)


def _manager_unavailable() -> HTTPException:
    return _http_error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "run manager is unavailable until the bounded v2 worker is configured",
    )


def _default_frontend_directory() -> Path | None:
    static = Path(__file__).with_name("static")
    return static if (static / "index.html").is_file() else None


def create_app(
    *,
    project_directory: str | Path = ".calc-flow-projects",
    checkpoint_directory: str | Path = ".calc-flow-checkpoints",
    project_store: ProjectStoreProtocol | None = None,
    checkpoint_store: CheckpointStoreProtocol | None = None,
    runtime: RuntimeProtocol | None = None,
    run_manager: RunManagerProtocol | None = None,
    frontend_directory: str | Path | None = None,
) -> FastAPI:
    """Create the local-only v2 API without opening a network listener."""
    projects = project_store or FileProjectStore(project_directory)
    checkpoints = checkpoint_store or FileCheckpointStore(checkpoint_directory)
    selected_runtime = runtime or Runtime()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if run_manager is not None:
            await run_in_threadpool(run_manager.shutdown)

    app = FastAPI(title="Calc Flow API", version="2.0.0a1", lifespan=lifespan)
    app.state.project_store = projects
    app.state.checkpoint_store = checkpoints
    app.state.runtime = selected_runtime
    app.state.run_manager = run_manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @app.get(f"{API_PREFIX}/catalog")
    def get_catalog() -> list[dict[str, object]]:
        return selected_runtime.catalog()

    @app.get(f"{API_PREFIX}/schema/project")
    def get_project_schema() -> dict[str, object]:
        schema = json.loads(project_json_schema())
        assert isinstance(schema, dict)
        return schema

    @app.get(f"{API_PREFIX}/projects", response_model=tuple[ProjectSummary, ...])
    async def list_projects() -> tuple[ProjectSummary, ...]:
        try:
            stored = await projects.list()
        except CalcFlowError as error:
            raise _native_error(error, operation="list") from error
        summaries = (_project_summary(project) for project in stored)
        return tuple(sorted(summaries, key=lambda item: item.id))

    @app.post(
        f"{API_PREFIX}/projects",
        response_model=ProjectDocument,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_project(request: ProjectCreateRequest) -> ProjectDocument:
        project = request.to_project()
        try:
            await projects.create(project)
        except CalcFlowError as error:
            raise _native_error(error, operation="create") from error
        return project

    @app.post(
        f"{API_PREFIX}/projects/import",
        response_model=ProjectDocument,
        status_code=status.HTTP_201_CREATED,
    )
    async def import_project(
        request: Request,
        format: str = Query(pattern="^(json|yaml)$"),
        replace: bool = False,
    ) -> ProjectDocument:
        content = await request.body()
        parser = import_project_json if format == "json" else import_project_yaml
        try:
            project = await run_in_threadpool(parser, content)
            if replace:
                await projects.put(project)
            else:
                await projects.create(project)
        except CalcFlowError as error:
            operation = "put" if replace else "create"
            raise _native_error(error, operation=operation) from error
        return project

    @app.get(f"{API_PREFIX}/projects/{{project_id}}", response_model=ProjectDocument)
    async def get_project(project_id: str) -> ProjectDocument:
        try:
            return await projects.get(project_id)
        except (CalcFlowError, KeyError) as error:
            raise _native_error(error, operation="get") from error

    @app.put(f"{API_PREFIX}/projects/{{project_id}}", response_model=ProjectDocument)
    async def put_project(
        project_id: str, request: ProjectCreateRequest
    ) -> ProjectDocument:
        project = request.to_project()
        if project.root["id"] != project_id:
            raise _http_error(
                status.HTTP_409_CONFLICT,
                "path project ID does not match the document",
            )
        try:
            await projects.put(project)
        except CalcFlowError as error:
            raise _native_error(error, operation="put") from error
        return project

    @app.delete(
        f"{API_PREFIX}/projects/{{project_id}}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_project(project_id: str) -> Response:
        try:
            await projects.delete(project_id)
        except (CalcFlowError, KeyError) as error:
            raise _native_error(error, operation="delete") from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/export")
    async def export_project(
        project_id: str,
        format: str = Query(default="json", pattern="^(json|yaml)$"),
    ) -> PlainTextResponse:
        try:
            project = await projects.get(project_id)
            serializer = (
                export_project_json if format == "json" else export_project_yaml
            )
            document = await run_in_threadpool(serializer, project)
        except (CalcFlowError, KeyError) as error:
            operation = "get" if "not found" in str(error) else "export"
            raise _native_error(error, operation=operation) from error
        media_type = "application/json" if format == "json" else "application/yaml"
        return PlainTextResponse(
            document,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{project_id}.{format}"'
            },
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/validate")
    async def validate_stored_project(project_id: str) -> dict[str, object]:
        try:
            project = await projects.get(project_id)
            report = await run_in_threadpool(
                selected_runtime.validation_report, project.canonical_json()
            )
        except (CalcFlowError, KeyError) as error:
            operation = "get" if "not found" in str(error) else "validate"
            raise _native_error(error, operation=operation) from error
        return report

    async def compiled_project(project_id: str) -> tuple[ProjectDocument, object]:
        try:
            project = await projects.get(project_id)
            plan = await run_in_threadpool(
                selected_runtime.compile_project, project.canonical_json()
            )
        except (CalcFlowError, KeyError) as error:
            operation = "get" if "not found" in str(error) else "compile"
            raise _native_error(error, operation=operation) from error
        return project, plan

    async def checkpoint_summary(project_id: str) -> CheckpointSummary:
        _, plan = await compiled_project(project_id)
        pipeline_name = str(plan.name)
        fingerprint = str(plan.fingerprint)
        try:
            checkpoint = await checkpoints.load(pipeline_name)
        except CalcFlowError as error:
            raise _native_error(error, operation="checkpoint") from error
        if checkpoint is None:
            return CheckpointSummary(pipeline_name=pipeline_name, exists=False)
        state = checkpoint.get("state", {})
        if not isinstance(state, dict):
            raise _http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "checkpoint state must be a JSON object",
            )
        return CheckpointSummary(
            pipeline_name=pipeline_name,
            exists=True,
            compatible=checkpoint.get("pipeline_fingerprint") == fingerprint,
            pipeline_fingerprint=str(checkpoint["pipeline_fingerprint"]),
            sequence=int(checkpoint["sequence"]),
            source_cursor=checkpoint.get("source_cursor"),
            created_at=checkpoint.get("created_at"),
            state_nodes=tuple(sorted(state)),
        )

    @app.get(
        f"{API_PREFIX}/projects/{{project_id}}/checkpoint",
        response_model=CheckpointSummary,
    )
    async def get_project_checkpoint(project_id: str) -> CheckpointSummary:
        return await checkpoint_summary(project_id)

    @app.delete(
        f"{API_PREFIX}/projects/{{project_id}}/checkpoint",
        response_model=CheckpointSummary,
    )
    async def delete_project_checkpoint(project_id: str) -> CheckpointSummary:
        _, plan = await compiled_project(project_id)
        pipeline_name = str(plan.name)
        try:
            await checkpoints.delete(pipeline_name)
        except CalcFlowError as error:
            raise _native_error(error, operation="checkpoint") from error
        return CheckpointSummary(pipeline_name=pipeline_name, exists=False)

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/runs",
        response_model=RunResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_run(project_id: str, request: RunRequest) -> RunResponse:
        try:
            project = await projects.get(project_id)
        except (CalcFlowError, KeyError) as error:
            raise _native_error(error, operation="get") from error
        if run_manager is None:
            raise _manager_unavailable()
        try:
            return await run_in_threadpool(run_manager.submit, project, request)
        except KeyError as error:
            raise _http_error(status.HTTP_404_NOT_FOUND, str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise _http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
            ) from error

    @app.get(f"{API_PREFIX}/runs/{{run_id}}", response_model=RunResponse)
    async def get_run(run_id: str) -> RunResponse:
        if run_manager is None:
            raise _manager_unavailable()
        try:
            return await run_in_threadpool(run_manager.get, run_id)
        except KeyError as error:
            raise _http_error(status.HTTP_404_NOT_FOUND, str(error)) from error

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/events")
    async def get_run_events(
        run_id: str,
        last_event_id: int | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        if run_manager is None:
            raise _manager_unavailable()
        try:
            await run_in_threadpool(run_manager.get, run_id)
        except KeyError as error:
            raise _http_error(status.HTTP_404_NOT_FOUND, str(error)) from error

        async def stream() -> AsyncIterator[str]:
            after_sequence = last_event_id if last_event_id is not None else -1
            terminal = {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.TIMED_OUT,
            }
            yield "retry: 500\n\n"
            while True:
                events, run_status = await run_in_threadpool(
                    run_manager.wait_for_events,
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
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete(f"{API_PREFIX}/runs/{{run_id}}", response_model=RunResponse)
    async def cancel_run(run_id: str) -> RunResponse:
        if run_manager is None:
            raise _manager_unavailable()
        try:
            return await run_in_threadpool(run_manager.cancel, run_id)
        except KeyError as error:
            raise _http_error(status.HTTP_404_NOT_FOUND, str(error)) from error

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
    """Run the unauthenticated v2 service on a loopback interface only."""
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
